#!/usr/bin/env python3
"""
Generates the 5-diagram sequence explaining the register/shared-memory
layout transformation across MM0 -> softmax -> MM0 epilogue -> MM1 prologue
-> MM1 accumulator in CUTLASS example 41 (fused multi-head attention),
for scalar_t=half, ArchTag=Sm80 (what SM89/Ada actually instantiates),
InstructionShape=16x8x8, WarpShape=32x32.

We render a representative 16x16 excerpt of the full 32x32 warp tile
(mma_m=0, mma_n in {0,1}) since it already contains full lane diversity
(quad 0-7 x lane_in_quad 0-3) and the pattern repeats identically for the
remaining mma_m/mma_n tiles.
"""
import sys
import os
sys.path.insert(0, os.path.expanduser(r"C:\Users\12587\.claude\skills\excalidraw-diagrams\scripts"))

from excalidraw_generator import Diagram, DiagramStyle, rectangle, text, COLORS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 8 distinct colors, one per "quad" (lane_id >> 2). Each quad = 4 lanes
# (lane_in_quad 0-3) that own the SAME row-band (row = quad, quad+8) but
# DIFFERENT columns. This directly matches the row-max/row-sum reduction
# structure shown in diagram 2.
QUAD_COLORS = ["red", "orange", "yellow", "green", "teal", "cyan", "blue", "violet"]
GROUP_COLORS = ["blue", "teal", "orange", "grape"]  # ldmatrix lane-groups of 8

CELL = 34
FONT = 12


def lane_at(row, col):
    """Inverse of AccumLambdaIteratorSm80 mapping, restricted to the
    representative window row in [0,16), col in [0,16) i.e. mma_m=0,
    mma_n in {0,1}, row_sub in {0,1}, col_sub in {0,1}."""
    quad = row % 8
    row_sub = row // 8          # 0 or 1  -> selects "row" in the formula
    mma_n = col // 8
    lane_in_quad = (col % 8) // 2
    col_sub = col % 2
    lane = quad * 4 + lane_in_quad
    return lane, quad, lane_in_quad, row_sub, mma_n, col_sub


def add_grid(d, ox, oy, color_of, label_of, cell=CELL, rows=16, cols=16, font_size=FONT):
    for r in range(rows):
        for c in range(cols):
            x = ox + c * cell
            y = oy + r * cell
            color = color_of(r, c)
            rect = rectangle(x, y, cell, cell, color=color, roughness=0, stroke_width=1)
            d.add(rect)
            label = label_of(r, c)
            if label:
                t = text(x, y + cell / 2 - font_size * 0.65, label,
                          font_size=font_size, font_family="code", color="black", align="center")
                t["width"] = cell
                t["textAlign"] = "center"
                d.add(t)


def add_row_col_axis(d, ox, oy, rows=16, cols=16, cell=CELL):
    for c in range(cols):
        t = text(ox + c * cell, oy - 20, str(c), font_size=11, font_family="code", color="gray", align="center")
        t["width"] = cell
        t["textAlign"] = "center"
        d.add(t)
    for r in range(rows):
        t = text(ox - 26, oy + r * cell + cell / 2 - 8, str(r), font_size=11, font_family="code", color="gray", align="right")
        t["width"] = 22
        t["textAlign"] = "right"
        d.add(t)


def legend(d, x, y):
    d.text_box(x, y, "Legend: quad = lane_id>>2 (0-7)   lane_in_quad = lane_id&3 (0-3)", font_size=14, color="black")
    d.text_box(x, y + 22, "warpSize=32, one warp owns one 32x32 accumulator tile (WarpShape=32x32).", font_size=13, color="gray")
    d.text_box(x, y + 42, "Threadblock (kQueriesPerBlock=kKeysPerBlock=64) = 4 such warps tiled 2x2.", font_size=13, color="gray")
    d.text_box(x, y + 62, "This excerpt shows rows 0-15, cols 0-15 (mma_m=0, mma_n in {0,1}) - pattern repeats for the rest of the 32x32 tile.", font_size=13, color="gray")


# ---------------------------------------------------------------------------
# Diagram 1: MM0 accumulator output (registers)
# ---------------------------------------------------------------------------
def diagram1():
    d = Diagram(diagram_style=DiagramStyle(roughness=0, stroke_width=1))
    d.text_box(20, 10, "Diagram 1/5 - MM0 accumulator output (registers)", font_size=24, color="black")
    d.text_box(20, 42, "P = Q @ K^T  |  m16n8k8 Sm80 TensorOp accumulator (MmaTensorOpAccumulatorTileIterator)", font_size=15, color="black")
    d.text_box(20, 64, "accum_m = mma_m*16 + row*8 + quad   |   accum_n = mma_n*8 + lane_in_quad*2 + col   (AccumLambdaIteratorSm80)", font_size=13, color="gray")

    ox, oy = 90, 140

    def color_of(r, c):
        lane, quad, *_ = lane_at(r, c)
        return QUAD_COLORS[quad % 8]

    def label_of(r, c):
        lane, *_ = lane_at(r, c)
        return f"L{lane}"

    add_row_col_axis(d, ox, oy)
    add_grid(d, ox, oy, color_of, label_of)
    d.text_box(ox, oy - 55, "accum_n (column, within 32x32 warp tile)", font_size=13, color="gray")
    d.text_box(ox - 90, oy + 250, "accum_m\n(row)", font_size=13, color="gray")

    d.text_box(ox + 16 * CELL + 40, oy, "Each lane owns 8 elements total:\n2 rows {quad, quad+8} x 4 cols\n{lane_in_quad*2+col, 8+lane_in_quad*2+col}.\n\n4 lanes with the SAME quad (same\ncolor) own the SAME row-band but\nDIFFERENT columns - this is exactly\nthe group that must cooperate in the\nsoftmax row reduction (Diagram 2).", font_size=13, color="black")

    legend(d, 20, 620)
    d.text_box(20, 700, "Instantiation note: on SM89(Ada), DISPATCH_ARCHTAG treats CC>=80 as arch::Sm80, so this is the exact\naccumulator layout used (m16n8k8 f16 tensor-op MMA, same as A100).", font_size=13, color="orange")
    d.save(os.path.join(OUT_DIR, "03-mha-mm0-accum-layout.excalidraw"))


# ---------------------------------------------------------------------------
# Diagram 2: Softmax reduction in registers + shared arrays
# ---------------------------------------------------------------------------
def diagram2():
    d = Diagram(diagram_style=DiagramStyle(roughness=0, stroke_width=1))
    d.text_box(20, 10, "Diagram 2/5 - softmax reduction: registers -> shared arrays", font_size=24, color="black")
    d.text_box(20, 42, "iterative_softmax() in kernel_forward.h - row-max via atomicMax, row-sum via butterfly shuffle", font_size=15, color="black")

    ox, oy = 90, 110

    def color_of(r, c):
        lane, quad, *_ = lane_at(r, c)
        return QUAD_COLORS[quad % 8]

    def label_of(r, c):
        lane, *_ = lane_at(r, c)
        return f"L{lane}"

    add_row_col_axis(d, ox, oy)
    add_grid(d, ox, oy, color_of, label_of)

    # (a) row-max reduction: highlight row = 3 (quad=3), 4 lanes lane_in_quad 0..3 -> lanes 12,13,14,15
    row_hl = 3
    mi_box = d.box(ox + 16 * CELL + 60, oy + row_hl * CELL - 10, "mi[accum_m=3]", color="red", width=170, height=48)
    for c in [0, 2, 4, 6]:  # one cell per lane_in_quad within mma_n=0
        cx = ox + c * CELL + CELL
        cy = oy + row_hl * CELL + CELL / 2
        from excalidraw_generator import Element
        src = Element(rectangle(cx - CELL, cy - CELL / 2, CELL, CELL), cx - CELL, cy - CELL / 2, CELL, CELL)
        d.arrow_between(src, mi_box, color="red", routing="straight")
    d.text_box(ox + 16 * CELL + 60, oy + row_hl * CELL + 45, "atomicMaxFloat(&mi[accum_m], max)\n4 lanes, same quad=3, different cols\n-> converge into ONE shared slot", font_size=12, color="red")

    # (b) row-sum reduction: quad=5 lanes 20,21,22,23, butterfly shuffle then addition_storage -> s_prime
    quad_b = 5
    lanes_b = [quad_b * 4 + k for k in range(4)]
    by = oy + 16 * CELL + 70
    bx = ox
    boxes = []
    for i, lane in enumerate(lanes_b):
        b = d.box(bx + i * 160, by, f"L{lane}\n(lane_in_quad={i})", color=QUAD_COLORS[quad_b], width=130, height=50)
        boxes.append(b)
    d.text_box(bx, by - 26, f"quad={quad_b} (row_band = {{{quad_b},{quad_b+8}}}) - butterfly shuffle reduceSameRow():", font_size=13, color="black")
    # xor 1: 0<->1, 2<->3
    d.arrow_between(boxes[0], boxes[1], "shfl_xor(1)", color="gray", routing="straight")
    d.arrow_between(boxes[2], boxes[3], "shfl_xor(1)", color="gray", routing="straight")
    # xor 2: 0<->2, 1<->3 (draw below)
    d.line_between(boxes[0], boxes[2], color="gray")
    d.line_between(boxes[1], boxes[3], color="gray")
    d.text_box(bx + 640, by + 5, "shfl_xor(2) (0<->2, 1<->3)", font_size=12, color="gray")

    add_box = d.box(bx + 60, by + 110, "addition_storage[\n  accum_m + kQueriesPerBlock*tile_offset.column()]", color="orange", width=340, height=60)
    d.arrow_between(boxes[0], add_box, "lane_in_quad==0 writes\ntotal_row", color="orange")

    s_prime_box = d.box(bx + 60, by + 210, "s_prime[row]\n(sum over MmaCore::WarpCount::kN partials)", color="green", width=340, height=60)
    d.arrow_between(add_box, s_prime_box, "thread_id<kQueriesPerBlock\nfinal accumulation loop", color="green")

    legend(d, 20, by + 300)
    d.save(os.path.join(OUT_DIR, "03-mha-softmax-reduction.excalidraw"))


# ---------------------------------------------------------------------------
# Diagram 3: MM0 epilogue - B2bGemm::accumToSmem (registers -> smem Si)
# ---------------------------------------------------------------------------
def diagram3():
    d = Diagram(diagram_style=DiagramStyle(roughness=0, stroke_width=1))
    d.text_box(20, 10, "Diagram 3/5 - MM0 epilogue: B2bGemm::accumToSmem (registers -> shared memory Si)", font_size=22, color="black")
    d.text_box(20, 42, "FragmentIteratorTensorOp -> TileIteratorTensorOp (ldsm-compatible smem write), cast accum_t(float) -> scalar_t(half)", font_size=14, color="black")

    # left: faded lane-colored register tile (small copy of diagram 1)
    ox1, oy1 = 60, 130

    def color_of(r, c):
        lane, quad, *_ = lane_at(r, c)
        return QUAD_COLORS[quad % 8]

    def label_of(r, c):
        lane, *_ = lane_at(r, c)
        return f"L{lane}"

    add_grid(d, ox1, oy1, color_of, label_of, cell=22, font_size=8)
    d.text_box(ox1, oy1 - 26, "registers (per-lane, faded)", font_size=13, color="gray")

    # right: plain row-major smem tile, no lane coloring
    ox2, oy2 = 560, 130

    def color_of2(r, c):
        return "gray"

    def label_of2(r, c):
        return f"({r},{c})"

    add_grid(d, ox2, oy2, color_of2, label_of2, cell=34, font_size=9)
    add_row_col_axis(d, ox2, oy2)
    d.text_box(ox2, oy2 - 55, "shared memory Si (AccumulatorSharedStorage) - row-major, addressable by (row,col) ONLY", font_size=13, color="black")

    from excalidraw_generator import Element
    left = Element(rectangle(ox1, oy1, 16 * 22, 16 * 22), ox1, oy1, 16 * 22, 16 * 22)
    right = Element(rectangle(ox2, oy2, 16 * 34, 16 * 34), ox2, oy2, 16 * 34, 16 * 34)
    d.arrow_between(left, right, "scatter\n(dashed)", color="violet", routing="straight")

    d.text_box(20, 640, "Key point: after accumToSmem(), the tile has NO per-lane identity anymore.\nIt is stored as the `si` union member of SharedStorageAfterMM0, and reused later as MM1 operand A (Pij @ V).", font_size=15, color="black")
    legend(d, 20, 700)
    d.save(os.path.join(OUT_DIR, "03-mha-epilogue-to-smem.excalidraw"))


# ---------------------------------------------------------------------------
# Diagram 4: MM1 prologue - WarpIteratorFromSmem ldmatrix load
# ---------------------------------------------------------------------------
def diagram4():
    d = Diagram(diagram_style=DiagramStyle(roughness=0, stroke_width=1))
    d.text_box(20, 10, "Diagram 4/5 - MM1 prologue: WarpIteratorFromSmem ldmatrix load (smem -> NEW registers)", font_size=21, color="black")
    d.text_box(20, 42, "Operand A for MM1's warp_mma (Operator1) - ldmatrix has its own FIXED hardware distribution pattern,", font_size=14, color="black")
    d.text_box(20, 62, "NOT the same per-lane mapping as Diagram 1.", font_size=14, color="black")

    # left: smem tile (same as diagram 3 right side)
    ox1, oy1 = 60, 140

    def color_of(r, c):
        return "gray"

    def label_of(r, c):
        return f"({r},{c})"

    add_grid(d, ox1, oy1, color_of, label_of, cell=32, font_size=9)
    add_row_col_axis(d, ox1, oy1, cell=32)
    d.text_box(ox1, oy1 - 55, "shared memory Si (row-major, from Diagram 3)", font_size=13, color="black")

    # right: ldmatrix loaded registers - each row r (0-15) is owned ENTIRELY by lane r
    ox2, oy2 = 620, 140

    def color_of2(r, c):
        lane = r  # row r owned entirely by lane r within this 16-row window
        group = lane // 8
        return GROUP_COLORS[group % 4]

    def label_of2(r, c):
        lane = r
        return f"L{lane}" if c == 7 else ""

    add_grid(d, ox2, oy2, color_of2, label_of2, cell=32, font_size=10)
    add_row_col_axis(d, ox2, oy2, cell=32)
    d.text_box(ox2, oy2 - 55, "operand-A registers after ldmatrix.sync.aligned.x4", font_size=13, color="black")

    from excalidraw_generator import Element
    left = Element(rectangle(ox1, oy1, 16 * 32, 16 * 32), ox1, oy1, 16 * 32, 16 * 32)
    right = Element(rectangle(ox2, oy2, 16 * 32, 16 * 32), ox2, oy2, 16 * 32, 16 * 32)
    d.arrow_between(left, right, "ldmatrix.x4\n(4 groups of 8 lanes)", color="blue", routing="straight")

    d.text_box(20, 680,
               "ldsm_vec_num = lane_id>>3 (0-3) = 4 lane-groups of 8 (group = lane_id div 8).\n"
               "Group 0 (lanes 0-7) loads rows 0-7 (access_m_idx=0); Group 1 (lanes 8-15) loads rows 8-15 (access_m_idx=1) - both from the SAME 16-row\n"
               "instruction tile (mma_m=0) shown here. Groups 2-3 (lanes 16-31, not shown) load rows 16-31 (mma_m=1, kTilesPerInstruction=2).\n"
               "Within this window each lane ends up owning one FULL 16-column row (origin=(lane%8,0), kElementsPerAccess=8 for half) via a single\n"
               "ldmatrix.sync.aligned.x4.m8n8.shared.b16 instruction per group of 8 lanes.",
               font_size=13, color="black")
    legend(d, 20, 800)
    d.save(os.path.join(OUT_DIR, "03-mha-mm1-prologue-ldmatrix.excalidraw"))


# ---------------------------------------------------------------------------
# Diagram 5: MM1 accumulator output (registers) - accum_o
# ---------------------------------------------------------------------------
def diagram5():
    d = Diagram(diagram_style=DiagramStyle(roughness=0, stroke_width=1))
    d.text_box(20, 10, "Diagram 5/5 - MM1 accumulator output (registers): accum_o", font_size=24, color="black")
    d.text_box(20, 42, "accum_o = P @ V  |  same m16n8k8 Sm80 TensorOp accumulator layout as Diagram 1 (same IteratorC kind)", font_size=15, color="black")
    d.text_box(20, 64, "shape: kQueriesPerBlock x head_dim_value  (32x32 warp tile shown here, excerpt rows/cols 0-15)", font_size=13, color="gray")

    ox, oy = 90, 140

    def color_of(r, c):
        lane, quad, *_ = lane_at(r, c)
        return QUAD_COLORS[quad % 8]

    def label_of(r, c):
        lane, *_ = lane_at(r, c)
        return f"L{lane}"

    add_row_col_axis(d, ox, oy)
    add_grid(d, ox, oy, color_of, label_of)
    d.text_box(ox, oy - 55, "accum_n (column, within head_dim_value)", font_size=13, color="gray")
    d.text_box(ox - 90, oy + 250, "accum_m\n(row = query)", font_size=13, color="gray")

    d.text_box(ox + 16 * CELL + 40, oy,
               "accum_o is consumed directly by:\n\n"
               "1) MemoryEfficientAttentionNormalize\n"
               "   epilogue - rescale by s_prime[row]\n"
               "   (and out_rescale if not first tile),\n"
               "   OR\n\n"
               "2) the NEXT key-tile's iterative_softmax\n"
               "   rescale-in-place step, if kKeepOutputInRF\n"
               "   is true and this is not the last key tile.",
               font_size=13, color="black")

    legend(d, 20, 620)
    d.save(os.path.join(OUT_DIR, "03-mha-mm1-accum-layout.excalidraw"))


if __name__ == "__main__":
    diagram1()
    diagram2()
    diagram3()
    diagram4()
    diagram5()
    print("done")
