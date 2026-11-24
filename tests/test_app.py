"""Web 层端到端自测：python tests/test_app.py。"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from werkzeug.datastructures import MultiDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, CASES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ✓", msg)


def png_bytes(gray):
    buf = io.BytesIO()
    Image.fromarray(gray.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def make_upload(client, images, lights, candidates):
    data = []
    for i, gray in enumerate(images):
        data.append(("image", (io.BytesIO(png_bytes(gray)),
                               f"img_{i}.png", "image/png")))
    for i, light in enumerate(lights):
        for c in range(3):
            data.append((f"light_{i}_{c}", str(light[c])))
    data.append(("candidate_count", str(len(candidates))))
    for j, cand in enumerate(candidates):
        for c in range(3):
            data.append((f"cand_{j}_{c}", str(cand[c])))
    return client.post("/upload", data=MultiDict(data),
                       content_type="multipart/form-data")


def base_tiles():
    meta = json.load(open(os.path.join(DEMO, "meta.json"), encoding="utf-8"))
    images = [
        np.asarray(Image.open(os.path.join(DEMO, name)).convert("L"),
                   dtype=np.uint8)
        for name in meta["images"]
    ]
    return images, meta


def test_index_and_demo(client):
    print("test_index_and_demo")
    resp = client.get("/")
    check(resp.status_code == 200 and "灯位".encode() in resp.data,
          "首页可打开")
    resp = client.get("/demo")
    check(resp.status_code == 200, "演示页可打开")
    # 演示数据在内存案例表中生成，取最新一条
    cid = next(iter(CASES.keys()))
    img = client.get(f"/case/{cid}/image/0")
    check(img.status_code == 200 and img.mimetype == "image/png",
          "裁片以 PNG 返回")
    check(client.get(f"/case/{cid}/image/99").status_code == 404,
          "越界图片索引返回 404")


def test_upload_validation(client):
    print("test_upload_validation")
    images, meta = base_tiles()

    # 非 PNG / 非灰度模式拒绝
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    data = [("image", (io.BytesIO(buf.getvalue()), "x.png", "image/png"))]
    for i in range(3):
        g = images[i]
        data.append(("image", (io.BytesIO(png_bytes(g)),
                               f"y{i}.png", "image/png")))
    for i in range(4):
        for c in range(3):
            data.append((f"light_{i}_{c}", str(meta["lights"][i][c])))
    data.append(("candidate_count", "2"))
    for j in range(2):
        for c in range(3):
            data.append((f"cand_{j}_{c}", str(meta["candidates"][j][c])))
    resp = client.post("/upload", data=MultiDict(data),
                       content_type="multipart/form-data")
    check(resp.status_code == 400 and "灰度" in resp.get_json()["error"],
          "RGB 图被拒绝并提示灰度要求")

    # 尺寸不一致拒绝
    mismatched = images[:3] + [np.zeros((3, 3), dtype=np.uint8)]
    resp = make_upload(client, mismatched, meta["lights"],
                       meta["candidates"][:2])
    check(resp.status_code == 400 and "尺寸" in resp.get_json()["error"],
          "尺寸不一致被拒绝")

    # 超过 16 像素拒绝
    big = [np.zeros((17, 4), dtype=np.uint8) for _ in range(4)]
    resp = make_upload(client, big, meta["lights"], meta["candidates"][:2])
    check(resp.status_code == 400 and "16" in resp.get_json()["error"],
          "每边超过 16 像素被拒绝")

    # 备选灯数量越界
    resp = make_upload(client, images, meta["lights"], [meta["candidates"][0]])
    check(resp.status_code == 400 and "备选灯" in resp.get_json()["error"],
          "备选灯只有 1 盏被拒绝")


def test_solve_flow_and_recompute(client):
    print("test_solve_flow_and_recompute")
    images, meta = base_tiles()
    resp = make_upload(client, images, meta["lights"], meta["candidates"])
    body = resp.get_json()
    check(resp.status_code == 200 and body["case_id"], "合法上传成功")
    cid = body["case_id"]

    pairs = meta["suggested_pairs"]
    payload = {"case_id": cid, "pairs": pairs,
               "reliable": [True] * 4}
    resp = client.post("/solve", json=payload)
    r = resp.get_json()
    check(resp.status_code == 200 and r["ok"], "求解成功")
    check(r["result"]["recommendation"] == 0
          and r["result"]["guarantee"] > 0,
          f"推荐第1盏、保证值 {r['result']['guarantee']}")
    qualified = [v["index"] for v in r["result"]["verdicts"]
                 if v["qualified"]]
    check(qualified == [0, 1, 3], "合格灯集合与演示设计一致")
    check(not r["result"]["verdicts"][2]["qualified"],
          "越界灯被判不合格")

    # 点对校验：两点相同拒绝
    bad = dict(payload, pairs=[[[1, 1], [1, 1]]])
    resp = client.post("/solve", json=bad)
    check(resp.status_code == 400 and "不同" in resp.get_json()["error"],
          "同像素点对被拒绝")

    # 全部不可靠拒绝
    bad = dict(payload, reliable=[False] * 4)
    resp = client.post("/solve", json=bad)
    check(resp.status_code == 400 and "可靠" in resp.get_json()["error"],
          "没有可靠照片时拒绝")

    # 改标重算：污染第一张并标为不可靠 → 排除它后仍应给出推荐
    case = CASES[cid]
    polluted = case["images"].copy()
    polluted[0] = np.clip(polluted[0].astype(int) + 40, 0, 255)
    case["images"] = polluted.astype(np.uint8)
    resp = client.post("/solve", json={
        "case_id": cid, "pairs": pairs,
        "reliable": [False, True, True, True]})
    r2 = resp.get_json()["result"]
    check("排除第 1 张" in r2["feasible_exclusions"][0],
          "改标后出现排除第1张的可行解释")
    check(r2["recommendation"] is not None, "排除污染照片后重新产生推荐")

    # 再把污染照片标回可靠 → 无共同解释，推荐撤下并提示
    resp = client.post("/solve", json={
        "case_id": cid, "pairs": pairs,
        "reliable": [True, True, True, True]})
    r3 = resp.get_json()["result"]
    check(r3["infeasible"] and r3["recommendation"] is None,
          "无共同解释时撤下推荐")


def test_no_usable_light(client):
    print("test_no_usable_light")
    # 构造可行原照，但所有备选灯都会越界 → 无灯可用提示
    rng = np.random.default_rng(7)
    lights = rng.normal(size=(4, 3))
    lights[:, 2] = np.abs(lights[:, 2]) + 1.0
    r_true = np.array([0.1, -0.2, 100.0])
    tiles = []
    for L in lights:
        g = int(round(float(L @ r_true)))
        tiles.append(np.full((3, 3), np.clip(g, 10, 240), dtype=np.uint8))
    candidates = [[5.0, 0.0, 3.0], [-5.0, 0.0, 3.0]]
    resp = make_upload(client, tiles, lights.tolist(), candidates)
    check(resp.status_code == 200, f"上传成功: {resp.data[:200]}")
    cid = resp.get_json()["case_id"]
    resp = client.post("/solve", json={
        "case_id": cid,
        "pairs": [[[0, 0], [1, 1]]],
        "reliable": [True] * 4})
    r = resp.get_json()["result"]
    check(not r["infeasible"], "原照本身存在共同解释")
    check(r["recommendation"] is None and all(
        not v["qualified"] for v in r["verdicts"]),
        "所有灯不合格 → 无灯可用且推荐撤下")


def shadow_demo_upload(client):
    """上传随仓库附带的阴影演示裁片，返回 (case_id, meta)。"""
    meta = json.load(open(os.path.join(DEMO, "meta.json"),
                          encoding="utf-8"))["shadow_demo"]
    images = [
        np.asarray(Image.open(os.path.join(DEMO, name)).convert("L"),
                   dtype=np.uint8)
        for name in meta["images"]
    ]
    resp = make_upload(client, images, meta["lights"], meta["candidates"])
    assert resp.status_code == 200, resp.data
    return resp.get_json()["case_id"], meta


def test_shadow_diagnose_enable_disable(client):
    print("test_shadow_diagnose_enable_disable")
    cid, meta = shadow_demo_upload(client)
    pairs = meta["suggested_pairs"]

    # 未诊断时不能启用
    resp = client.post("/shadow/enable", json={"case_id": cid})
    check(resp.status_code == 400 and "诊断" in resp.get_json()["error"],
          "未诊断时启用被拒绝")

    # 未启用掩膜：原照无共同解释，推荐撤下
    resp = client.post("/solve", json={
        "case_id": cid, "pairs": pairs, "reliable": [True] * 4})
    r0 = resp.get_json()["result"]
    check(r0["infeasible"] and r0["recommendation"] is None,
          "未启用掩膜时无共同解释、推荐撤下")
    check(r0["shadow"]["source"] is None, "结果标记当前未启用掩膜")

    # 诊断：最短边界 4、唯一掩膜、左上 2x2、无分歧
    resp = client.post("/shadow/diagnose", json={"case_id": cid, "source": 0})
    body = resp.get_json()
    check(resp.status_code == 200, "阴影诊断成功")
    d = body["diagnosis"]
    check(not d["conflict"] and d["unique"], "诊断无冲突且最优掩膜唯一")
    check(d["boundary"] == 4, f"最短边界 4，实际 {d['boundary']}")
    check(sorted(map(tuple, d["all_shadow"])) ==
          [(0, 0), (0, 1), (1, 0), (1, 1)], "所有最优掩膜都含左上 2x2")
    check(d["ambiguous"] == [], "无判定有分歧的像素")
    check(d["shadow_margin"] == 3, "阴影像素单边裕量为 3")

    # 原照索引越界拒绝
    resp = client.post("/shadow/diagnose", json={"case_id": cid, "source": 9})
    check(resp.status_code == 400, "诊断原照索引越界被拒绝")

    # 启用：按当前点对重算，给出推荐；单案例同时只有一个掩膜
    resp = client.post("/shadow/enable", json={
        "case_id": cid, "pairs": pairs, "reliable": [True] * 4})
    body = resp.get_json()
    check(resp.status_code == 200, "启用掩膜成功")
    check(body["shadow"]["active"]["source"] == 0
          and body["shadow"]["active"]["count"] == 4,
          "启用的掩膜覆盖第 1 张原照的 4 个像素")
    r1 = body["result"]
    check(not r1["infeasible"], "启用后存在共同解释")
    check(r1["recommendation"] == 0,
          f"启用后推荐第 1 盏，实际 {r1['recommendation']}")
    check(r1["guarantee"] is not None and r1["guarantee"] > 0,
          f"推荐保证值为正 ({r1['guarantee']})")
    check(r1["shadow"]["source"] == 0 and r1["shadow"]["count"] == 4,
          "求解结果标记当前启用的掩膜")

    # 再次求解（不带掩膜字段）仍沿用已启用掩膜
    resp = client.post("/solve", json={
        "case_id": cid, "pairs": pairs, "reliable": [True] * 4})
    r2 = resp.get_json()["result"]
    check(r2["recommendation"] == 0 and not r2["infeasible"],
          "重新求解沿用已启用掩膜")
    check(CASES[cid]["shadow_mask"] is not None, "案例中记录已启用掩膜")

    # 撤销：恢复到未启用状态，无共同解释、推荐撤下
    resp = client.post("/shadow/disable", json={"case_id": cid})
    body = resp.get_json()
    check(resp.status_code == 200 and body["shadow"]["active"] is None,
          "撤销掩膜成功")
    r3 = body["result"]
    check(r3["infeasible"] and r3["recommendation"] is None,
          "撤销后恢复无共同解释并撤下推荐")
    check(CASES[cid]["shadow_mask"] is None, "案例中掩膜已清除")


def test_shadow_ambiguity_and_conflict(client):
    print("test_shadow_ambiguity_and_conflict")
    # 四张同灰度、照明向量较平缓：每个像素在“阴影/非阴影”下都可行，
    # 空掩膜与整张阴影的边界同为 0，最优掩膜不唯一，禁止启用。
    one = np.full((3, 3), 100, dtype=np.uint8)
    lights = [[0.1864, 0.0067, 0.0527], [0.1715, 0.1588, 0.2611],
              [0.0936, -0.1881, 0.1267], [0.0575, -0.174, 0.2368]]
    images = [one, one, one, one]
    resp = make_upload(client, images, lights,
                       [[0, 0, 0.1], [0.02, 0.01, 0.1]])
    check(resp.status_code == 200, "歧义用例上传成功")
    cid = resp.get_json()["case_id"]
    resp = client.post("/shadow/diagnose", json={"case_id": cid, "source": 0})
    d = resp.get_json()["diagnosis"]
    check(not d["conflict"], "同灰度同灯位时无冲突")
    check(d["boundary"] == 0 and not d["unique"],
          f"全自由像素：边界 0、掩膜不唯一，实际 boundary={d['boundary']}")
    check(len(d["ambiguous"]) == 9, "9 个像素都判定有分歧")
    resp = client.post("/shadow/enable", json={"case_id": cid})
    check(resp.status_code == 400 and "唯一" in resp.get_json()["error"],
          "最优掩膜不唯一时启用被拒绝")

    # 观测冲突：整张原照偏亮 40 灰度，两种假设都不可行
    cid, meta = shadow_demo_upload(client)
    case = CASES[cid]
    broken = case["images"].copy()
    broken[0] = np.clip(broken[0].astype(int) + 40, 0, 255).astype(np.uint8)
    case["images"] = broken
    resp = client.post("/shadow/diagnose", json={"case_id": cid, "source": 0})
    d = resp.get_json()["diagnosis"]
    check(d["conflict"] and d["conflict_pixels"], "诊断报观测冲突并给出冲突像素")
    check(not d["unique"] and d["all_shadow"] == [], "观测冲突时没有可行掩膜")
    resp = client.post("/shadow/enable", json={"case_id": cid})
    check(resp.status_code == 400 and "冲突" in resp.get_json()["error"],
          "观测冲突时启用被拒绝")


def test_shadow_solve_error_keeps_state(client):
    print("test_shadow_solve_error_keeps_state")
    import app as app_module
    cid, meta = shadow_demo_upload(client)
    pairs = meta["suggested_pairs"]
    client.post("/shadow/diagnose", json={"case_id": cid, "source": 0})
    resp = client.post("/shadow/enable", json={
        "case_id": cid, "pairs": pairs, "reliable": [True] * 4})
    before = resp.get_json()
    old_mask = CASES[cid]["shadow_mask"]
    old_last = CASES[cid]["last_result"]
    check(before["result"]["recommendation"] == 0, "启用后先得到推荐")

    real_evaluate = app_module.evaluate

    def boom(_scene):
        raise RuntimeError("注入的求解故障")

    app_module.evaluate = boom
    try:
        resp = client.post("/shadow/disable", json={"case_id": cid})
        check(resp.status_code == 500 and "原设置" in resp.get_json()["error"],
              "求解出错时返回错误并说明已保留原设置")
        check(CASES[cid]["shadow_mask"] == old_mask, "掩膜未被撤销")
        check(CASES[cid]["last_result"] is old_last, "旧推荐未被覆盖")
    finally:
        app_module.evaluate = real_evaluate

    # 恢复后正常撤销
    resp = client.post("/shadow/disable", json={"case_id": cid})
    check(resp.status_code == 200
          and resp.get_json()["shadow"]["active"] is None,
          "故障解除后可正常撤销")


def main():
    app.config["TESTING"] = True
    CASES.clear()
    with app.test_client() as client:
        test_index_and_demo(client)
        test_upload_validation(client)
        test_solve_flow_and_recompute(client)
        test_no_usable_light(client)
        test_shadow_diagnose_enable_disable(client)
        test_shadow_ambiguity_and_conflict(client)
        test_shadow_solve_error_keeps_state(client)
    print("\nWeb 层全部自测通过。")


if __name__ == "__main__":
    main()
