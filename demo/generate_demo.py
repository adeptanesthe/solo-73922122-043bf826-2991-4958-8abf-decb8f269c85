"""生成演示用灰度裁片与备选灯元数据。

模型与求解器一致：灰度 g = L · r（取整），其中 r=(x,y,z)，z≥0。
默认演示（4x4，4 张原照）包含两类像素：
* 底面像素：反射向量接近 (0, 0, 240)；
* 刻线像素：带有横向/纵向倾斜，点积随灯位变化更明显。
生成结果全部落在 [5, 250]，避免 PNG 截断。

另生成一组“支架投影压暗一角”的 5x5 演示（shadow_*.png）：
* 第 1 张原照左上 2x2 被压暗 4 个灰度，其余像素与照片不变；
* 阴影诊断在全部原照参与下得到最短边界 4、唯一最优掩膜的左上 2x2；
* 未启用掩膜时原照无共同解释，启用后第 1 盏灯可被推荐。
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

WIDTH = HEIGHT = 4

# 每像素反射向量 (x, y, z)，z≥0
REFLECTORS = {
    "base": np.array([0.0, 0.0, 240.0]),
    "line_x": np.array([120.0, 0.0, 180.0]),
    "line_y": np.array([0.0, 120.0, 180.0]),
}

# 4 张原照的照明向量
LIGHTS = [
    [0.25, 0.10, 0.85],
    [-0.20, 0.30, 0.90],
    [0.15, -0.25, 0.80],
    [-0.10, -0.15, 0.95],
]

# 像素布局：B=底面，X=x 向刻线，Y=y 向刻线
LAYOUT = [
    ["B", "B", "B", "B"],
    ["B", "X", "Y", "B"],
    ["B", "B", "B", "B"],
    ["B", "X", "Y", "B"],
]

# 备选灯（数值经 LP 可行域实测标定）：
# 1) 合格、正保证，且保证值最大（预期被推荐）
# 2) 合格但保证值为负（刻线极值可反超底面，不推荐）
# 3) 不合格：部分像素预测灰度越过 [5,250]
# 4) 合格、正保证但保证值低于第 1 盏（次优，用于验证排序）
CANDIDATES = [
    [0.00, 0.00, 0.90],
    [0.70, 0.00, 0.70],
    [1.50, 0.00, 0.90],
    [0.10, 0.10, 0.80],
]

# ---- 阴影演示（5x5，4 张原照）-------------------------------------------
SHADOW_SIZE = 5
SHADOW_LIGHTS = [
    [0.114, -0.121, 0.587],
    [0.721, -0.325, 0.746],
    [0.353, -0.335, 0.798],
    [-0.429, 0.362, 0.563],
]
# 反射体：B=底面，L=刻线（x 向倾斜），S=阴影下的同一底面
SHADOW_REFLECTORS = {
    "B": np.array([0.0, 0.0, 240.0]),
    "L": np.array([110.0, 0.0, 180.0]),
    "S": np.array([0.0, 0.0, 240.0]),
}
SHADOW_LAYOUT = [
    ["S", "S", "B", "B", "B"],
    ["S", "S", "L", "B", "B"],
    ["B", "B", "B", "B", "B"],
    ["B", "B", "B", "B", "B"],
    ["B", "B", "B", "B", "B"],
]
SHADOW_CELLS = {(0, 0), (1, 0), (0, 1), (1, 1)}  # 第 1 张原照被压暗的像素
SHADOW_DROP = 4  # 压暗多少灰度

SHADOW_CANDIDATES = [
    [0.0, 0.0, 0.8],   # 合格、正保证，预期推荐
    [0.5, 0.0, 0.8],   # 合格但保证值为负
    [1.5, 0.0, 0.9],   # 不合格：预测灰度越界
    [0.1, 0.1, 1.0],   # 合格、正保证但低于第 1 盏
]


def main():
    reflector_grid = np.zeros((HEIGHT, WIDTH, 3))
    for y, row in enumerate(LAYOUT):
        for x, kind in enumerate(row):
            key = {"B": "base", "X": "line_x", "Y": "line_y"}[kind]
            reflector_grid[y, x] = REFLECTORS[key]

    images = []
    for i, light in enumerate(LIGHTS):
        light = np.asarray(light)
        raw = reflector_grid @ light
        gray = np.rint(raw).astype(np.uint8)
        lo, hi = int(gray.min()), int(gray.max())
        assert 5 <= lo and hi <= 250, f"演示图 {i} 灰度越界: {lo}..{hi}"
        images.append(gray)
        Image.fromarray(gray, mode="L").save(
            os.path.join(HERE, f"demo_{i}.png"))

    meta = {
        "size": [WIDTH, HEIGHT],
        "images": [f"demo_{i}.png" for i in range(len(LIGHTS))],
        "lights": LIGHTS,
        "candidates": CANDIDATES,
        # 与 LAYOUT 对应的建议点对（刻线, 底面）
        "suggested_pairs": [
            [[1, 1], [0, 0]],
            [[2, 1], [3, 1]],
        ],
        "layout": LAYOUT,
    }
    with open(os.path.join(HERE, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # 打印理论范围，方便人工核对
    print("候选灯在各像素类型上的理论灰度 L·r：")
    for j, cand in enumerate(CANDIDATES):
        cand = np.asarray(cand)
        vals = {name: float(r @ cand) for name, r in REFLECTORS.items()}
        print(f"  灯{j + 1} {cand.tolist()}: " +
              ", ".join(f"{k}={v:.1f}" for k, v in vals.items()))

    generate_shadow_demo()


def generate_shadow_demo():
    """生成 5x5 阴影演示裁片，并把元数据并入 meta.json。"""
    lights = np.asarray(SHADOW_LIGHTS)
    grid = np.zeros((SHADOW_SIZE, SHADOW_SIZE, 3))
    for y, row in enumerate(SHADOW_LAYOUT):
        for x, kind in enumerate(row):
            grid[y, x] = SHADOW_REFLECTORS[kind]

    images = []
    for i, light in enumerate(lights):
        gray = np.rint(grid @ light).astype(int)
        assert 0 <= gray.min() and gray.max() <= 255
        if i == 0:
            for (x, y) in SHADOW_CELLS:
                gray[y, x] -= SHADOW_DROP
        assert 0 <= gray.min() and gray.max() <= 255
        images.append(gray.astype(np.uint8))
        Image.fromarray(gray.astype(np.uint8), mode="L").save(
            os.path.join(HERE, f"shadow_{i}.png"))

    meta_path = os.path.join(HERE, "meta.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["shadow_demo"] = {
        "size": [SHADOW_SIZE, SHADOW_SIZE],
        "images": [f"shadow_{i}.png" for i in range(len(lights))],
        "lights": SHADOW_LIGHTS,
        "candidates": SHADOW_CANDIDATES,
        "suggested_pairs": [[[2, 1], [4, 4]]],
        "shadow_source": 0,
        "layout": SHADOW_LAYOUT,
        "note": ("第 1 张原照左上 2x2 被支架投影压暗；全组原照参与诊断，"
                 "最短边界 4 条边，唯一最优掩膜。"),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("已生成阴影演示 shadow_*.png 与 meta.json 的 shadow_demo 段。")


if __name__ == "__main__":
    main()
