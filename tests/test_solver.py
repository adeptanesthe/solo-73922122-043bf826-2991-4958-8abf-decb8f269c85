"""solver 单元自测（不依赖 Web 层），直接 python tests/test_solver.py 运行。"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import (GRAY_TOL, SHADOW_MARGIN, Scene, diagnose_shadow,
                    evaluate)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")


def load_demo(reliable_override=None):
    with open(os.path.join(DEMO, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    images = np.stack([
        np.asarray(Image.open(os.path.join(DEMO, name)).convert("L"),
                   dtype=np.uint8)
        for name in meta["images"]
    ])
    reliable = (reliable_override if reliable_override is not None
                else [True] * len(meta["images"]))
    return Scene(
        images=images,
        lights=np.asarray(meta["lights"], dtype=np.float64),
        candidates=np.asarray(meta["candidates"], dtype=np.float64),
        pairs=[(tuple(p[0]), tuple(p[1])) for p in meta["suggested_pairs"]],
        reliable=reliable,
    ), meta


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ✓", msg)


def test_demo_recommendation():
    print("test_demo_recommendation")
    scene, meta = load_demo()
    result = evaluate(scene)
    check(not result["infeasible"], "存在可行解释")
    check(result["feasible_exclusions"] == [None],
          "全部可靠时唯一可行解释为不排除任何照片")
    qualified = [v["index"] for v in result["verdicts"] if v["qualified"]]
    check(qualified == [0, 1, 3], f"合格灯为第1、2、4盏，实际 {qualified}")
    check(result["recommendation"] == 0, "推荐第 1 盏")
    check(result["guarantee"] is not None and result["guarantee"] > 0,
          f"推荐保证值为正: {result['guarantee']}")
    g1 = result["verdicts"][0]["guarantee"]
    g4 = result["verdicts"][3]["guarantee"]
    check(g1 > g4 > 0, f"第1盏保证值 {g1} 大于第4盏 {g4}")
    g2 = result["verdicts"][1]["guarantee"]
    check(g2 <= 0, f"第2盏合格但保证值非正 ({g2})，不得被推荐")
    v3 = result["verdicts"][2]
    check(not v3["qualified"] and v3["fail_reason"],
          "第3盏不合格并给出原因")


def test_tolerance_band():
    print("test_tolerance_band")
    scene, _ = load_demo()
    # 把某张原照整体偏移 2：仍应可行（边界恰好在 ±2）；偏移 3 则不可行
    scene.images[0] = np.clip(
        scene.images[0].astype(int) + 2, 0, 255).astype(np.uint8)
    result = evaluate(scene)
    check(not result["infeasible"], "灰度整体 +2 仍在容差内，可行")

    scene2, _ = load_demo()
    scene2.images[0] = np.clip(
        scene2.images[0].astype(int) + 30, 0, 255).astype(np.uint8)
    result2 = evaluate(scene2)
    check(result2["infeasible"], "灰度整体 +30 超出可行松弛，且全部可靠 → 无共同解释")


def test_exclusion_and_reliable():
    print("test_exclusion_and_reliable")
    scene, _ = load_demo()
    # 污染第 1 张（index 0），并把它标为不可靠：排除它应恢复可行
    scene.images[0] = np.clip(
        scene.images[0].astype(int) + 40, 0, 255).astype(np.uint8)
    scene.reliable = [False, True, True, True]
    result = evaluate(scene)
    check(not result["infeasible"], "允许排除污染照片后存在可行解释")
    check(result["feasible_exclusions"] == [0],
          f"唯一可行解释是排除第1张，实际 {result['feasible_exclusions']}")

    # 污染照片标为可靠 → 无法排除 → 无共同解释
    scene.reliable = [True, True, True, True]
    result_reliable = evaluate(scene)
    check(result_reliable["infeasible"],
          "污染照片被标为可靠时无共同解释")
    check(all(v["guarantee"] is None for v in result_reliable["verdicts"]),
          "无共同解释时不给出任何灯位保证")
    check(result_reliable["recommendation"] is None, "推荐撤下")


def test_z_nonnegative_enforced():
    print("test_z_nonnegative_enforced")
    # 单像素、2x2 裁片；构造只有负 z 才能满足的约束：
    # 两张灯分别让点积目标都偏高，而 x/y 分量无法兼顾——用随机方向验证
    rng = np.random.default_rng(0)
    lights = rng.normal(size=(4, 3))
    lights[:, 2] = np.abs(lights[:, 2]) + 0.5
    # 用一个 z>=0 的真实反射体生成数据，保证可行；再反向检查 z 约束生效
    r_true = np.array([0.2, -0.3, 100.0])
    images = np.stack([
        np.full((2, 2), np.clip(round(float(L @ r_true)), 0, 255),
                dtype=np.uint8)
        for L in lights
    ])
    scene = Scene(images=images, lights=lights,
                  candidates=np.zeros((2, 3)),
                  pairs=[((0, 0), (1, 1))], reliable=[True] * 4)
    result = evaluate(scene)
    check(not result["infeasible"], "z≥0 真实反射体数据可行")

    # 零向量灯：预测恒为 0，落在 [5,250] 之外 → 不合格
    check(not result["verdicts"][0]["qualified"],
          "零向量备选灯预测灰度 0，不合格")


def test_tie_prefers_first():
    print("test_tie_prefers_first")
    scene, _ = load_demo()
    # 两盏完全相同的灯 → 同分，按输入顺序取首盏
    scene.candidates = np.tile(scene.candidates[0], (2, 1))
    result = evaluate(scene)
    check(result["recommendation"] == 0, "同分推荐输入顺序首盏")
    check(result["verdicts"][0]["guarantee"]
          == result["verdicts"][1]["guarantee"], "两盏保证值相同")


def test_pair_difference_must_be_positive():
    print("test_pair_difference_must_be_positive")
    scene, _ = load_demo()
    result = evaluate(scene)
    cand2 = result["verdicts"][1]
    check(cand2["qualified"] and cand2["guarantee"] <= 0,
          f"第2盏合格但保证值非正 ({cand2['guarantee']})，不得推荐")
    check(result["recommendation"] != 1, "负保证灯未被推荐")


def load_shadow_demo():
    with open(os.path.join(DEMO, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)["shadow_demo"]
    images = np.stack([
        np.asarray(Image.open(os.path.join(DEMO, name)).convert("L"),
                   dtype=np.uint8)
        for name in meta["images"]
    ])
    return images, np.asarray(meta["lights"], dtype=np.float64), meta


def test_shadow_diagnosis():
    print("test_shadow_diagnosis")
    images, lights, meta = load_shadow_demo()
    diag = diagnose_shadow(images, lights, 0)
    check(not diag["conflict"], "阴影诊断无观测冲突")
    check(diag["boundary"] == 4,
          f"左上 2x2 与其余像素之间边界为 4 条边，实际 {diag['boundary']}")
    check(sorted(diag["all_shadow"]) == [(0, 0), (0, 1), (1, 0), (1, 1)],
          f"所有最优掩膜都判为阴影的像素是左上 2x2，实际 {diag['all_shadow']}")
    check(diag["ambiguous"] == [], "没有判定有分歧的像素")
    check(diag["unique"], "唯一最优掩膜")
    check(SHADOW_MARGIN == 3.0, "阴影像素单边裕量为 3")

    # 未启用掩膜：原照无共同解释
    pairs = [(tuple(p[0]), tuple(p[1])) for p in meta["suggested_pairs"]]
    plain = Scene(images=images, lights=lights,
                  candidates=np.asarray(meta["candidates"]),
                  pairs=pairs, reliable=[True] * 4)
    check(evaluate(plain)["infeasible"], "未启用掩膜时原照无共同解释")

    # 启用掩膜：按单边约束重算，并给出推荐
    masked = Scene(images=images, lights=lights,
                   candidates=np.asarray(meta["candidates"]),
                   pairs=pairs, reliable=[True] * 4,
                   shadow_source=0,
                   shadow_pixels=frozenset(diag["all_shadow"]))
    result = evaluate(masked)
    check(not result["infeasible"], "启用掩膜后存在共同解释")
    check(result["feasible_exclusions"] == [None],
          "启用掩膜且全部可靠时唯一解释为不排除")
    check(result["recommendation"] == 0,
          f"启用掩膜后推荐第 1 盏，实际 {result['recommendation']}")
    check(result["guarantee"] is not None and result["guarantee"] > 0,
          f"推荐保证值为正 ({result['guarantee']})")

    # 只把部分阴影像素改成单边约束会破坏可行域，说明单边约束确实生效
    partial = Scene(images=images, lights=lights,
                    candidates=np.asarray(meta["candidates"]),
                    pairs=pairs, reliable=[True] * 4,
                    shadow_source=0,
                    shadow_pixels=frozenset({(0, 0), (0, 1)}) - {(0, 1)})
    partial_result = evaluate(partial)
    check(partial_result["infeasible"], "掩膜缺一角时观测互相冲突")


def test_shadow_conflict_and_ambiguity():
    print("test_shadow_conflict_and_ambiguity")
    images, lights, _ = load_shadow_demo()

    # 冲突：把第 1 张原照整张 +40，任何标签都无法同时满足全组观测。
    broken = images.copy()
    broken[0] = np.clip(broken[0].astype(int) + 40, 0, 255).astype(np.uint8)
    diag = diagnose_shadow(broken, lights, 0)
    check(diag["conflict"], "整张偏亮导致观测冲突")
    check(diag["conflict_pixels"], "冲突像素非空")
    check(diag["mask"] is None and not diag["unique"],
          "观测冲突时没有可行掩膜")

    # 歧义：只有一个原照时，每个像素在“阴影/非阴影”下都可行，
    # 空掩膜与全阴影掩膜的边界同为 0，没有任何像素被所有最优掩膜判为阴影。
    one_image = np.full((1, 3, 3), 100, dtype=np.uint8)
    one_light = np.array([[0.2, 0.1, 0.9]])
    diag2 = diagnose_shadow(one_image, one_light, 0)
    check(not diag2["conflict"], "单原照不存在观测冲突")
    check(diag2["boundary"] == 0, "全自由像素时最短边界为 0")
    check(len(diag2["ambiguous"]) == 9 and diag2["all_shadow"] == [],
          "全部像素都判定有分歧")
    check(not diag2["unique"], "最优掩膜不唯一，不能启用")

    # 注入强制标签直接验证最小割：同一行两端强制阴影、下方强制非阴影，
    # 中间的自由像素既可连成一整条阴影带，也可只留两端，边界同为 4。
    from solver import _min_boundary_labels
    kinds = [2] * 16
    kinds[0] = 1        # (0,0) 强制阴影
    kinds[3] = 1        # (3,0) 强制阴影
    for x in range(4):  # 第二行强制非阴影
        kinds[4 + x] = 0
    boundary, all_shadow, ambiguous = _min_boundary_labels(kinds, 4, 4)
    check(boundary == 4, f"最短边界为 4，实际 {boundary}")
    check(sorted(all_shadow) == [(0, 0), (3, 0)],
          f"两端是所有最优掩膜的共同阴影，实际 {all_shadow}")
    check(sorted(ambiguous) == [(1, 0), (2, 0)],
          f"顶行中间像素判定有分歧，实际 {ambiguous}")

    # 一整块强制阴影、其余全部强制非阴影：边界最短且唯一。
    kinds2 = [0] * 16
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        kinds2[y * 4 + x] = 1
    boundary2, all2, amb2 = _min_boundary_labels(kinds2, 4, 4)
    check(boundary2 == 4 and not amb2,
          f"左上 2x2 强制阴影、其余强制非阴影时边界 4、无分歧，"
          f"实际 boundary={boundary2} amb={amb2}")


def test_shadow_mask_scope():
    print("test_shadow_mask_scope")
    images, lights, meta = load_shadow_demo()
    pairs = [((2, 1), (4, 4))]
    # 掩膜只对该原照生效：标为其他原照时并不改善可行性。
    wrong = Scene(images=images, lights=lights,
                  candidates=np.asarray(meta["candidates"]),
                  pairs=pairs, reliable=[True] * 4,
                  shadow_source=3,
                  shadow_pixels=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}))
    check(evaluate(wrong)["infeasible"], "掩膜只作用于所选原照，标错原照仍不可行")

    # 启用掩膜不改变“排除整张原照”的原有规则：第 1 张仍可被排除。
    masked = Scene(images=images, lights=lights,
                   candidates=np.asarray(meta["candidates"]),
                   pairs=pairs, reliable=[False, True, True, True],
                   shadow_source=0,
                   shadow_pixels=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}))
    result = evaluate(masked)
    check(0 in result["feasible_exclusions"],
          f"启用掩膜后仍允许按原规则排除整张原照，实际 {result['feasible_exclusions']}")


def test_shadow_diagnosis_max_crop():
    print("test_shadow_diagnosis_max_crop")
    # 覆盖允许的最大裁片尺寸 16x16：最短边界与掩膜仍由一次最小割给出。
    rng = np.random.default_rng(4)
    k, h, w = 8, 16, 16
    lights = np.column_stack([rng.uniform(-0.5, 0.5, (k, 2)),
                              rng.uniform(0.6, 1.0, k)])
    grid = np.zeros((h, w, 3))
    grid[:, :, 2] = 200.0
    for (x, y) in [(3, 3), (4, 3), (7, 8), (8, 8), (11, 5)]:
        grid[y, x] = [100.0, 0.0, 150.0]
    raw = np.rint(grid @ lights.T).astype(int)
    dark = {(x, y) for x in range(5) for y in range(4)}
    for (x, y) in dark:
        raw[y, x, 0] -= 6
    images = np.clip(raw, 0, 255).astype(np.uint8).transpose(2, 0, 1)
    diag = diagnose_shadow(images, lights, 0)
    check(not diag["conflict"], "16x16 诊断无冲突")
    check(diag["unique"], "16x16 最优掩膜唯一")
    check(set(diag["all_shadow"]) == dark,
          f"16x16 掩膜精确覆盖被压暗的 5x4 区域，实际 {len(diag['all_shadow'])} 像素")
    # 边界 = 5x4 区域外露的边数：右侧 4 + 下侧 5 = 9
    check(diag["boundary"] == 9, f"16x16 最短边界 9，实际 {diag['boundary']}")


if __name__ == "__main__":
    test_demo_recommendation()
    test_tolerance_band()
    test_exclusion_and_reliable()
    test_z_nonnegative_enforced()
    test_tie_prefers_first()
    test_pair_difference_must_be_positive()
    test_shadow_diagnosis()
    test_shadow_conflict_and_ambiguity()
    test_shadow_mask_scope()
    test_shadow_diagnosis_max_crop()
    print("\n全部自测通过。")
