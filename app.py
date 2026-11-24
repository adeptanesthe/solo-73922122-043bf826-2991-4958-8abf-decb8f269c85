"""石刻浅刻线补拍灯位选择 —— Flask Web 应用。

流程：
1. 上传 4~8 张已对齐的灰度 PNG 裁片（同尺寸、每边至多 16 像素），
   填写每张原照的三维照明向量与 2~8 个备选灯向量；
2. 在放大裁片上点选 1~4 对（刻线像素, 底面像素），两点必须不同；
3. 后端用 Pillow 读图、SciPy(HiGHS) 求线性规划，给出合格灯与推荐；
4. 更改“可靠”标记后重新求解，推荐随结果更新或撤下。
"""
from __future__ import annotations

import io
import json
import math
import os
import time
import uuid

import numpy as np
from flask import (Flask, jsonify, render_template, request,
                   send_file, session)
from PIL import Image

from solver import (GRAY_HIGH, GRAY_LOW, GRAY_TOL, SHADOW_MARGIN, Scene,
                    diagnose_shadow, evaluate)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# 单用户工具：案例保存在进程内存中，键为随机案例 id。
CASES: dict = {}

MIN_IMAGES, MAX_IMAGES = 4, 8
MIN_CANDIDATES, MAX_CANDIDATES = 2, 8
MIN_PAIRS, MAX_PAIRS = 1, 4
MAX_SIDE = 16


# --------------------------------------------------------------------------
# 校验与解析
# --------------------------------------------------------------------------
class ValidationError(Exception):
    pass


def _parse_vec3(values, label):
    if len(values) != 3:
        raise ValidationError(f"{label} 需要 3 个分量")
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValidationError(f"{label} 含非法数字: {v!r}")
        if not math.isfinite(f):
            raise ValidationError(f"{label} 含非有限数值")
        out.append(f)
    return out


def _load_grayscale_png(file_storage, label):
    """用 Pillow 读取，强制要求 PNG 格式、灰度模式。"""
    if file_storage is None or not file_storage.filename:
        raise ValidationError(f"{label} 缺少文件")
    if not file_storage.filename.lower().endswith(".png"):
        raise ValidationError(f"{label} 必须是 PNG 文件")
    try:
        img = Image.open(io.BytesIO(file_storage.read()))
        img.load()
    except Exception as exc:  # Pillow 会抛各类 OSError
        raise ValidationError(f"{label} 无法作为图片读取: {exc}")
    if img.format != "PNG":
        raise ValidationError(f"{label} 不是 PNG 格式")
    if img.mode != "L":
        raise ValidationError(
            f"{label} 必须是灰度图（模式 L），当前为 {img.mode}")
    if img.width < 1 or img.height < 1:
        raise ValidationError(f"{label} 尺寸非法")
    if img.width > MAX_SIDE or img.height > MAX_SIDE:
        raise ValidationError(
            f"{label} 每边至多 {MAX_SIDE} 像素，"
            f"当前 {img.width}x{img.height}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return np.asarray(img, dtype=np.uint8), buf.getvalue(), img.size


def _validate_pairs(pairs, width, height):
    if not isinstance(pairs, list) or not (MIN_PAIRS <= len(pairs) <= MAX_PAIRS):
        raise ValidationError(
            f"点对数量须在 {MIN_PAIRS}~{MAX_PAIRS} 之间")
    cleaned = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValidationError(f"第 {idx + 1} 对点结构非法")
        points = []
        for point in pair:
            if not isinstance(point, list) or len(point) != 2:
                raise ValidationError(f"第 {idx + 1} 对点坐标非法")
            x, y = point
            if (not isinstance(x, int) or not isinstance(y, int)
                    or not (0 <= x < width) or not (0 <= y < height)):
                raise ValidationError(
                    f"第 {idx + 1} 对点坐标 ({x},{y}) 超出裁片范围")
            points.append((x, y))
        if points[0] == points[1]:
            raise ValidationError(f"第 {idx + 1} 对点的两个像素必须不同")
        cleaned.append((points[0], points[1]))
    return cleaned


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        min_images=MIN_IMAGES, max_images=MAX_IMAGES,
        min_candidates=MIN_CANDIDATES, max_candidates=MAX_CANDIDATES,
        max_side=MAX_SIDE,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("image")
    if not (MIN_IMAGES <= len(files) <= MAX_IMAGES):
        raise ValidationError(
            f"原照数量须在 {MIN_IMAGES}~{MAX_IMAGES} 之间，当前 {len(files)}")

    arrays, pngs, sizes = [], [], []
    for i, fs in enumerate(files):
        arr, png, size = _load_grayscale_png(fs, f"第 {i + 1} 张原照")
        arrays.append(arr)
        pngs.append(png)
        sizes.append(size)
    if len(set(sizes)) != 1:
        raise ValidationError(f"所有裁片尺寸必须相同，当前分别为 {sizes}")
    width, height = sizes[0]

    lights = []
    for i in range(len(files)):
        lights.append(_parse_vec3(
            [request.form.get(f"light_{i}_{c}") for c in range(3)],
            f"第 {i + 1} 张原照的照明向量"))

    candidate_count = _int_field(request.form.get("candidate_count"),
                                 "备选灯数量")
    if not (MIN_CANDIDATES <= candidate_count <= MAX_CANDIDATES):
        raise ValidationError(
            f"备选灯数量须在 {MIN_CANDIDATES}~{MAX_CANDIDATES} 之间")
    candidates = []
    for j in range(candidate_count):
        candidates.append(_parse_vec3(
            [request.form.get(f"cand_{j}_{c}") for c in range(3)],
            f"第 {j + 1} 盏备选灯向量"))

    case_id = uuid.uuid4().hex
    CASES[case_id] = {
        "images": np.stack(arrays),
        "pngs": pngs,
        "size": (width, height),
        "lights": np.asarray(lights, dtype=np.float64),
        "candidates": np.asarray(candidates, dtype=np.float64),
        "created": time.time(),
        "last_result": None,
        # 阴影诊断结果与当前启用的掩膜（每案例至多一个）
        "shadow_diagnosis": None,
        "shadow_mask": None,
    }
    session["case_id"] = case_id
    return jsonify({"case_id": case_id, "width": width, "height": height,
                    "image_count": len(files),
                    "candidate_count": candidate_count})


@app.route("/demo")
def demo():
    """载入随仓库附带的演示裁片，进入同一套点选/求解界面。"""
    variant = "shadow" if request.args.get("scene") == "shadow" else "default"
    meta_path = os.path.join(app.root_path, "demo", "meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    if variant == "shadow":
        scene_meta = meta["shadow_demo"]
    else:
        scene_meta = meta

    arrays, pngs = [], []
    for name in scene_meta["images"]:
        path = os.path.join(app.root_path, "demo", name)
        with open(path, "rb") as fh:
            data = fh.read()
        img = Image.open(io.BytesIO(data))
        img.load()
        arrays.append(np.asarray(img.convert("L"), dtype=np.uint8))
        pngs.append(data)

    case_id = uuid.uuid4().hex
    CASES[case_id] = {
        "images": np.stack(arrays),
        "pngs": pngs,
        "size": tuple(scene_meta["size"]),
        "lights": np.asarray(scene_meta["lights"], dtype=np.float64),
        "candidates": np.asarray(scene_meta["candidates"], dtype=np.float64),
        "created": time.time(),
        "last_result": None,
        "demo_meta": scene_meta,
        "demo_variant": variant,
        "shadow_diagnosis": None,
        "shadow_mask": None,
    }
    session["case_id"] = case_id
    return render_template("case.html", case_id=case_id,
                           demo=scene_meta, demo_variant=variant,
                           max_pairs=MAX_PAIRS)


@app.route("/case/<case_id>/image/<int:index>")
def case_image(case_id, index):
    case = CASES.get(case_id)
    if case is None or not (0 <= index < len(case["pngs"])):
        return "not found", 404
    return send_file(io.BytesIO(case["pngs"][index]), mimetype="image/png")


@app.route("/case/<case_id>")
def case_view(case_id):
    case = CASES.get(case_id)
    if case is None:
        return render_template("case.html", case_id=None), 404
    return render_template("case.html", case_id=case_id, demo=None,
                           demo_variant="default", max_pairs=MAX_PAIRS)


@app.route("/solve", methods=["POST"])
def solve():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON")

    case = CASES.get(payload.get("case_id"))
    if case is None:
        raise ValidationError("案例不存在或已过期，请重新上传")

    width, height = case["size"]
    k = case["images"].shape[0]
    pairs = _validate_pairs(payload.get("pairs"), width, height)
    reliable = _validate_reliable(payload.get("reliable"), k)

    try:
        result = _run_evaluation(case, pairs, reliable)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": f"求解器错误: {exc}"}), 500

    # 以本次结果覆盖旧结果——推荐随最新标记/点对更新或撤下。
    case["last_result"] = {"pairs": pairs, "reliable": reliable,
                           "result": result}

    return jsonify({"ok": True, "result": _serialize_result(result, case)})


def _validate_reliable(reliable, k):
    if (not isinstance(reliable, list) or len(reliable) != k
            or not all(isinstance(v, bool) for v in reliable)):
        raise ValidationError("可靠标记格式非法")
    if not any(reliable):
        raise ValidationError("至少保留一张可靠原照")
    return reliable


def _active_mask(case):
    """当前启用的阴影掩膜 (source, frozenset) 或 None。"""
    mask = case.get("shadow_mask")
    if not mask:
        return None
    return mask["source"], frozenset(mask["pixels"])


def _run_evaluation(case, pairs, reliable):
    active = _active_mask(case)
    scene = Scene(
        images=case["images"],
        lights=case["lights"],
        candidates=case["candidates"],
        pairs=pairs,
        reliable=reliable,
        shadow_source=active[0] if active else None,
        shadow_pixels=active[1] if active else None,
    )
    return evaluate(scene)


# --------------------------------------------------------------------------
# 阴影诊断：同一案例至多启用一张掩膜
# --------------------------------------------------------------------------
def _shadow_state(case):
    active = case.get("shadow_mask")
    diag = case.get("shadow_diagnosis")
    return {
        "active": None if active is None else {
            "source": active["source"],
            "pixels": [list(p) for p in sorted(active["pixels"])],
            "count": len(active["pixels"]),
        },
        "diagnosis": None if diag is None else _serialize_diagnosis(diag),
    }


def _serialize_diagnosis(diag):
    return {
        "source": diag["source"],
        "conflict": diag["conflict"],
        "conflict_pixels": [list(p) for p in diag["conflict_pixels"]],
        "unique": diag["unique"],
        "boundary": diag["boundary"],
        "all_shadow": [list(p) for p in sorted(diag["all_shadow"])],
        "ambiguous": [list(p) for p in sorted(diag["ambiguous"])],
        "shadow_margin": SHADOW_MARGIN,
    }


@app.route("/shadow/diagnose", methods=["POST"])
def shadow_diagnose():
    """对选定原照做阴影诊断，返回最优掩膜与分歧像素。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON")
    case = CASES.get(payload.get("case_id"))
    if case is None:
        raise ValidationError("案例不存在或已过期，请重新上传")

    source = payload.get("source")
    k = case["images"].shape[0]
    if (not isinstance(source, int) or isinstance(source, bool)
            or not (0 <= source < k)):
        raise ValidationError("阴影诊断的原照索引非法")

    try:
        diag = diagnose_shadow(case["images"], case["lights"], source)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": f"求解器错误: {exc}"}), 500

    case["shadow_diagnosis"] = diag
    return jsonify({"ok": True, "diagnosis": _serialize_diagnosis(diag),
                    "shadow": _shadow_state(case)})


@app.route("/shadow/enable", methods=["POST"])
def shadow_enable():
    """启用唯一最优掩膜；按当前点对重算，失败则保留旧设置与推荐。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON")
    case = CASES.get(payload.get("case_id"))
    if case is None:
        raise ValidationError("案例不存在或已过期，请重新上传")

    diag = case.get("shadow_diagnosis")
    if diag is None:
        raise ValidationError("请先对一张原照做阴影诊断")
    if diag["conflict"]:
        raise ValidationError("该原照观测冲突，没有可行掩膜")
    if not diag["unique"]:
        raise ValidationError("最优掩膜不唯一，不能启用")

    source = diag["source"]
    if (isinstance(payload.get("source"), int)
            and payload["source"] != source):
        raise ValidationError("诊断结果与所选原照不一致，请重新诊断")

    width, height = case["size"]
    pairs, reliable = _pairs_reliable_for_recompute(case, payload,
                                                    width, height)
    mask = {"source": source, "pixels": [tuple(p) for p in diag["all_shadow"]]}

    old_mask = case.get("shadow_mask")
    case["shadow_mask"] = mask
    try:
        result = _run_evaluation(case, pairs, reliable)
    except RuntimeError as exc:
        # 求解出错：恢复旧设置与旧推荐。
        case["shadow_mask"] = old_mask
        return jsonify({"ok": False,
                        "error": f"求解器错误: {exc}，已保留原设置"}), 500

    case["last_result"] = {"pairs": pairs, "reliable": reliable,
                           "result": result}
    return jsonify({"ok": True, "result": _serialize_result(result, case),
                    "shadow": _shadow_state(case)})


@app.route("/shadow/disable", methods=["POST"])
def shadow_disable():
    """撤销当前掩膜；按当前点对重算，失败则保留旧设置与推荐。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON")
    case = CASES.get(payload.get("case_id"))
    if case is None:
        raise ValidationError("案例不存在或已过期，请重新上传")

    width, height = case["size"]
    pairs, reliable = _pairs_reliable_for_recompute(case, payload,
                                                    width, height)

    old_mask = case.get("shadow_mask")
    case["shadow_mask"] = None
    try:
        result = _run_evaluation(case, pairs, reliable)
    except RuntimeError as exc:
        case["shadow_mask"] = old_mask
        return jsonify({"ok": False,
                        "error": f"求解器错误: {exc}，已保留原设置"}), 500

    case["last_result"] = {"pairs": pairs, "reliable": reliable,
                           "result": result}
    return jsonify({"ok": True, "result": _serialize_result(result, case),
                    "shadow": _shadow_state(case)})


def _pairs_reliable_for_recompute(case, payload, width, height):
    """优先使用请求中的当前点对/可靠标记，缺省沿用上次求解输入。"""
    last = case.get("last_result") or {}
    pairs_raw = payload.get("pairs")
    if pairs_raw is None and last.get("pairs") is not None:
        # 上次求解存的是清洗后的元组，转回 JSON 形态再走同一套校验。
        pairs_raw = [[list(p[0]), list(p[1])] for p in last["pairs"]]
    reliable_raw = payload.get("reliable", last.get("reliable"))
    if pairs_raw is None or reliable_raw is None:
        raise ValidationError("请先完成一次求解，或随请求提交点对与可靠标记")
    pairs = _validate_pairs(pairs_raw, width, height)
    reliable = _validate_reliable(reliable_raw,
                                  case["images"].shape[0])
    return pairs, reliable


def _int_field(raw, label):
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"{label} 非法: {raw!r}")


def _serialize_result(result, case):
    width, _ = case["size"]
    exclusions = [
        ("不排除" if e is None else f"排除第 {e + 1} 张原照")
        for e in result["feasible_exclusions"]
    ]
    verdicts = []
    for v in result["verdicts"]:
        verdicts.append({
            "index": v["index"],
            "vector": [round(float(x), 4) for x in case["candidates"][v["index"]]],
            "qualified": v["qualified"],
            "guarantee": (None if v["guarantee"] is None
                          else round(v["guarantee"], 4)),
            "global_min": (None if v["global_min"] is None
                           else round(v["global_min"], 4)),
            "global_max": (None if v["global_max"] is None
                           else round(v["global_max"], 4)),
            "fail_reason": v["fail_reason"],
        })
    return {
        "infeasible": result["infeasible"],
        "feasible_exclusions": exclusions,
        "verdicts": verdicts,
        "recommendation": result["recommendation"],
        "guarantee": (None if result["guarantee"] is None
                      else round(result["guarantee"], 4)),
        "gray_band": [GRAY_LOW, GRAY_HIGH],
        "gray_tol": GRAY_TOL,
        "shadow": {
            "source": (None if not case.get("shadow_mask")
                       else case["shadow_mask"]["source"]),
            "count": (0 if not case.get("shadow_mask")
                      else len(case["shadow_mask"]["pixels"])),
            "margin": SHADOW_MARGIN,
        },
    }


@app.errorhandler(ValidationError)
def handle_validation(err):
    return jsonify({"ok": False, "error": str(err)}), 400


@app.errorhandler(413)
def handle_too_large(_):
    return jsonify({"ok": False, "error": "上传内容超过 8MB 限制"}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
