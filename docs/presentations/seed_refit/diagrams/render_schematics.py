"""Render compact, code-native SVG artwork; no algorithm or data is executed.

Run with the project's pytorch Python, from any working directory.
Mermaid sources describe the same control flow; *_detailed.svg are auto-layout
exports, while the compact primary diagrams below use explicit slide placement.
"""
from pathlib import Path
from html import escape
import math

OUT = Path(__file__).resolve().parents[1] / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
INK = '#23343D'
TEAL = '#E8F3F1'
BLUE = '#EDF0FA'
GOLD = '#FAF0D9'
RED = '#FAE9E5'
GREY = '#EEF1F2'
# Muted palette keeps geometry identical while dropping a stage to the background.
MUTE_FILL = '#FBFCFC'
MUTE_STROKE = '#DCE3E6'
MUTE_TEXT = '#B4C0C6'


class Canvas:
    def __init__(self, title, subtitle='', width=1600, height=900, conclusion='', badge='',
                 label_size=22, label_bold=False):
        self.w, self.h = width, height
        # Edge-label styling is a canvas-wide choice: the original single-page
        # figures keep 22 px plain labels, the stage figures use larger bold
        # ones so 是/否 stays readable from the back of the room.
        self.label_size, self.label_bold = label_size, label_bold
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8 Z" fill="#53616D"/></marker>'
                      '<marker id="arrowmute" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8 Z" fill="#C6D0D5"/></marker>'
                      f'<marker id="arrowink" markerWidth="11" markerHeight="9" refX="10" refY="4.5" orient="auto"><path d="M0 0 L11 4.5 L0 9 Z" fill="{INK}"/></marker></defs>',
                      '<rect width="100%" height="100%" fill="white"/>']
        # Only lift the title when a conclusion line has to fit beneath it.
        # The band figures carry their title in the slide itself, so an empty
        # one draws nothing rather than an empty text node.
        if title:
            self.text(65, 52 if conclusion else 58, title, 34, bold=True, anchor='start')
        if conclusion:
            self.text(65, 96, '结论：' + conclusion, 26, bold=True, anchor='start', color='#2E6B5E')
        elif subtitle:
            self.text(65, 98, subtitle, 22, anchor='start', color='#53616D')
        if badge:
            self.text(width-60, 52, badge, 26, bold=True, anchor='end', color='#8A97A0')

    def text(self, x, y, lines, size=25, bold=False, anchor='middle', color=INK):
        if isinstance(lines, str):
            lines = lines.split('\n')
        for j, line in enumerate(lines):
            content = escape(line).replace('s∩', 's<tspan baseline-shift="super" font-size="70%">∩</tspan>')
            self.parts.append(f'<text x="{x}" y="{y+j*(size*1.45)}" font-family="Droid Sans Fallback, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{content}</text>')

    def box(self, x, y, w, h, lines, fill=TEAL, size=25, mute=False, dash=False):
        f = MUTE_FILL if mute else fill
        stroke = MUTE_STROKE if mute else '#62777F'
        dash_attr = ' stroke-dasharray="10 7"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{f}" stroke="{stroke}" stroke-width="2"{dash_attr}/>')
        n = len(lines.split('\n'))
        self.text(x+w/2, y+h/2-(n-1)*size*1.45/2+size*.32, lines, size, color=MUTE_TEXT if mute else INK)

    def diamond(self, cx, cy, w, h, lines, size=23, mute=False):
        f = MUTE_FILL if mute else GOLD
        stroke = MUTE_STROKE if mute else '#9B8051'
        self.parts.append(f'<path d="M{cx-w/2} {cy} L{cx} {cy-h/2} L{cx+w/2} {cy} L{cx} {cy+h/2} Z" fill="{f}" stroke="{stroke}" stroke-width="2"/>')
        n = len(lines.split('\n'))
        self.text(cx, cy-(n-1)*size*1.45/2+size*.32, lines, size, color=MUTE_TEXT if mute else INK)

    def line(self, points, label=None, lx=None, ly=None, arrow=True, color='#53616D', dash=False, mute=False):
        col = '#C6D0D5' if mute else color
        d = 'M' + ' L'.join(f'{x},{y}' for x,y in points)
        dash_attr = 'stroke-dasharray="7 6"' if dash else ''
        arrow_attr = ('marker-end="url(#arrowmute)"' if mute else 'marker-end="url(#arrow)"') if arrow else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4" {dash_attr} {arrow_attr}/>')
        if label:
            self.text(lx, ly, label, self.label_size, color=col, bold=self.label_bold and not mute)

    def path(self, d, color='#287A6D', width=4, fill='none', dash=False):
        dash_attr = 'stroke-dasharray="8 7"' if dash else ''
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" {dash_attr}/>')

    def curve(self, d, label=None, lx=None, ly=None, size=22, arrow=True, color=INK):
        """Bezier edge for the band figures.

        The band diagrams have no node boxes: the curve itself is what carries
        the reading order, so it uses the ink-coloured arrow head and a heavier
        stroke than the boxed flowcharts' connectors.
        """
        arrow_attr = 'marker-end="url(#arrowink)"' if arrow else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" {arrow_attr}/>')
        if label:
            self.text(lx, ly, label, size, color=color)

    def dot(self, x, y, fill='#287A6D', r=7):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

    def save(self, name):
        (OUT / f'{name}.svg').write_text('\n'.join(self.parts)+'</svg>', encoding='utf-8')


def seed():
    c=Canvas('SMT seed｜几何预拟合', '非目标 SMT 点；五维优化；输出六个物理参数', height=1000)
    c.box(70,150,440,85,'Helix 初值\n规范固定与曲率缩放')
    c.box(70,295,440,105,'最近点 → 残差 → J、H、g\n逐径迹 decrement')
    c.line([(290,235),(290,295)])
    c.diamond(290,515,340,130,'分块内\n全部驻点？')
    c.line([(290,400),(290,450)])
    c.box(790,150,580,85,'未驻点径迹：求 LM 步并构造候选')
    c.box(790,290,580,95,'重新计算候选最近点\n目标函数及有效性')
    c.line([(460,515),(625,515),(625,192),(790,192)],'否',590,488)
    c.line([(1080,235),(1080,290)])
    c.diamond(1080,480,410,140,'逐径迹：有限且下降\n保护初始有效点？')
    c.line([(1080,385),(1080,410)])
    c.box(740,600,305,95,'提交候选\n阻尼 ×0.1，有下限')
    c.box(1120,600,305,95,'保留参数\n阻尼 ×10',RED)
    c.line([(875,480),(705,480),(705,565),(892,565),(892,600)],'是',850,550)
    c.line([(1285,480),(1320,480),(1320,600)],'否',1360,568)
    c.diamond(1080,815,310,115,'预算耗尽？')
    c.line([(892,695),(892,725),(1080,725)],arrow=False)
    c.line([(1272,695),(1272,725),(1080,725)],arrow=False)
    c.line([(1080,725),(1080,757)])
    c.line([(1235,815),(1500,815),(1500,130),(560,130),(560,347),(510,347)],'否：重新线性化',1100,117)
    c.box(70,635,440,95,'重新计算终态诊断\n驻点、非有限值与失效率',GOLD)
    c.line([(290,580),(290,635)],'是',255,614)
    c.line([(925,815),(580,815),(580,683),(510,683)],'是',745,798)
    c.diamond(290,825,300,100,'主线检查通过？',22)
    c.line([(290,730),(290,775)])
    c.box(65,915,410,65,'恢复六参数 → measurement refit',size=22)
    c.line([(290,875),(290,915)],'是',250,904)
    c.box(550,875,330,65,'失败退出并记录原因',RED,22)
    c.line([(440,825),(550,825),(550,907)],'否',490,808)
    c.save('p3_seed')


def refit():
    c=Canvas('Measurement refit｜局部测量重拟合', '固定工作几何；非目标 measurement；终态验收后固定径迹', height=1000)
    c.box(65,150,460,100,'SMT seed 与非目标 measurement\n初始求交检查 → 规范固定',BLUE)
    c.box(65,310,460,110,'求交 → 局部标准化残差\n构造 J、H、g',BLUE)
    c.line([(295,250),(295,310)],'通过',240,288)
    c.box(800,150,590,95,'求解 LM 步，构造候选\n重新求交并计算候选目标',BLUE)
    c.line([(525,365),(650,365),(650,197),(800,197)])
    c.diamond(1095,365,430,125,'逐径迹：候选目标\n严格下降？')
    c.line([(1095,245),(1095,302)])
    c.box(760,490,305,90,'提交候选\n阻尼 ×0.1，有下限',BLUE)
    c.box(1120,490,305,90,'保留参数\n阻尼 ×10',RED)
    c.line([(880,365),(725,365),(725,460),(912,460),(912,490)],'是',850,446)
    c.line([(1310,365),(1335,365),(1335,490)],'否',1370,461)
    c.box(845,640,500,80,'计算分块内最大 change',BLUE)
    c.line([(912,580),(912,610),(1095,610)],arrow=False)
    c.line([(1272,580),(1272,610),(1095,610)],arrow=False)
    c.line([(1095,610),(1095,640)])
    c.diamond(1095,825,435,135,'change < 10⁻¹⁰\n或预算耗尽？')
    c.line([(1095,720),(1095,757)])
    c.line([(1312,825),(1505,825),(1505,130),(560,130),(560,280),(295,280),(295,310)],'否：重新线性化',1120,117)
    c.box(65,540,460,115,'独立终态验收\ndecrement、非驻点比例\n非有限值与求交失效率',GOLD,23)
    c.line([(878,825),(630,825),(630,597),(525,597)],'是',733,807)
    c.diamond(295,745,330,110,'主线验收通过？')
    c.line([(295,655),(295,690)])
    c.box(65,865,450,75,'恢复六参数 → 固定径迹',BLUE)
    c.line([(295,800),(295,865)],'是',253,840)
    c.box(540,915,330,65,'失败退出并记录原因',RED,22)
    c.line([(460,745),(565,745),(565,915)],'否',513,730)
    c.line([(65,200),(30,200),(30,975),(540,975),(540,947)],'初始检查失败',285,970)
    c.save('p4_refit')


def blocklm_band():
    """P7 band figure, drawn in the same hand-flow style as the seed/refit bands.

    One main chain plus one return arc: the per-sensor blocking lives in the
    node wording and the damping ladder on the return arc's label, because the
    band form has no second nesting level to spend on them.  Every node
    corresponds to a step in ``run_block_lm_alignment``; the four terminal
    states stay as a caption under the single check node rather than fanning
    out into four exits, which would break the band's width.
    """
    c = Canvas('', width=1600, height=520)
    # Two inputs converging from the left, as in the seed band.
    c.text(45, 85, '固定径迹 helix\n（已通过驻点验收）', 24, anchor='start')
    c.text(45, 300, '目标层局部测量\n及测量方差', 24, anchor='start')
    c.curve('M272 98 C322 110 330 152 306 186')
    c.curve('M222 306 C268 300 288 272 306 244')
    # Main chain.
    c.text(430, 210, '逐 target sensor\nhelix 与该面求交', 26)
    c.text(760, 60, '转换为局部坐标系\n得到 LOC0／LOC1 预测', 26)
    c.text(1090, 60, '构造按方差\n标准化的残差', 26)
    c.text(1150, 210, '阻尼 LM\n调整该面对齐增量 δ', 26)
    c.text(1440, 210, '驻点与终态检查', 26)
    c.text(1440, 310, '输出逐 sensor\n对齐增量 δ', 24)
    c.curve('M442 178 C512 172 540 126 588 106')
    c.curve('M948 78 C972 66 988 62 1004 62')
    c.curve('M1128 112 C1146 146 1148 164 1150 178')
    c.curve('M1282 210 L1336 210')
    c.curve('M1440 232 L1440 282')
    # Return arc: the candidate is scored after moving this sensor's surface,
    # always on the component set the current point already had.
    c.curve('M1150 262 C1122 348 902 358 790 358 C654 358 462 352 434 270')
    c.text(790, 396, '按 δ 移动测量面 → 重求交 → 在固定分量集上评分', 23)
    c.text(790, 428, '接受：μ×0.5；拒绝：μ×10 重试，同一 Jacobian 复用', 21,
           color='#53616D')
    c.text(1580, 372, '终态：converged ／ rank_deficient\nbudget_exhausted ／ intersection_failed',
           18, anchor='end', color='#53616D')
    c.text(45, 470, '· 拟合过程中径迹固定，仅更新各 target sensor 的对齐增量；block 之间精确解耦。',
           22, anchor='start')
    c.text(45, 502, '· 迭代停止不等于收敛：converged 需通过独立的自由集驻点判据。',
           22, anchor='start')
    c.save('p7_blocklm_band')


def gauge():
    c=Canvas('规范固定｜同一条曲线的局部唯一表示', '示意图：展示参数原点冗余，不表示具体探测器几何',height=850)
    c.text(390,165,'同一 helix，可选择不同参数原点',27,bold=True)
    c.path('M110 530 C210 210 410 220 530 420 S690 660 765 350')
    c.dot(266.48,295.44);c.text(220,267,'原点 O',25)
    c.dot(530,420,fill='#9A7134');c.text(587,400,'原点 O′',25)
    c.line([(266.48,295.44),(355,249.5)],arrow=True,color='#53616D')
    c.line([(530,420),(578,500)],arrow=True,color='#53616D')
    c.text(410,610,'原点沿曲线移动，方位角同步改变',23)
    c.text(410,655,'几何轨迹不变：一个表示冗余',25,bold=True)
    c.text(1190,165,'固定参考法平面',27,bold=True)
    c.path('M920 530 L1130 230 L1480 340 L1270 640 Z',color='#9CADB4',width=2,fill=TEAL)
    c.line([(1170,445),(1370,390)],color='#53616D')
    c.text(1400,383,'e₀',25)
    c.line([(1170,445),(1080,573)],color='#53616D')
    c.text(1060,608,'e₁',25)
    c.line([(1170,445),(1130,210)],color='#9A7134')
    c.text(1250,210,'参考单位切向 d',24)
    c.dot(1170,445);c.text(1190,490,'r_ref',23)
    c.dot(1290,458,fill='#344A9B');c.text(1370,505,'优化原点 r₀',24)
    c.line([(1170,445),(1290,458)],arrow=False,dash=True)
    c.text(1190,700,'r₀ = r_ref + a₀e₀ + a₁e₁',28)
    c.box(200,755,1200,65,'六个物理参数 → 五个优化变量；曲率缩放与降维是两项不同处理',size=26)
    c.save('p2_gauge')


def closest():
    c=Canvas('最近点与测量面交点：两种不同的路径定义', '概念示意；具体路径均由当前 helix 参数数值求解',height=850)
    c.text(390,170,'SMT seed：点到曲线的最近点',27,bold=True)
    c.path('M100 640 C240 520 210 430 400 430 C560 430 640 490 730 330')
    c.dot(400,240,fill='#9A7134');c.text(455,220,'SMT 点 p',25)
    c.dot(400,430);c.line([(400,240),(400,430)],arrow=False,dash=True)
    c.line([(280,430),(540,430)],color='#53616D')
    c.text(440,478,'h(s*)',25);c.text(570,412,'切向 t',23)
    c.path('M400 405 L425 405 L425 430',width=2,color='#53616D')
    c.text(390,645,'内部精确最近点满足：',24)
    c.text(390,695,'[h(s*) − p] · t(s*) = 0',28)
    c.text(1190,170,'Measurement refit：曲线与平面求交',27,bold=True)
    c.path('M940 560 L1050 300 L1460 350 L1350 610 Z',width=2,color='#9CADB4',fill=BLUE)
    c.path('M900 650 C1080 600 1000 500 1180 450 S1350 280 1490 240',color='#344A9B')
    c.dot(1180,450,fill='#344A9B');c.text(1280,475,'交点 h(s∩)',25)
    c.dot(1070,490,fill='#53616D');c.text(1035,530,'中心 c',23)
    c.line([(1070,490),(1010,210)],color='#9A7134');c.text(985,220,'法向 n',25)
    c.text(1190,695,'[h(s∩) − c] · n = 0',28)
    c.box(190,755,1220,65,'最近点约束是距离驻点条件；求交约束是预测点位于测量平面',size=26)
    c.save('a6_closest_vs_intersection')


def intersection():
    c=Canvas('从全局交点到局部测量预测', '几何固定；R 将局部坐标映射到全局，Rᵀ 用于逆变换',height=850)
    c.path('M140 580 L300 260 L760 350 L600 670 Z',width=2,color='#9CADB4',fill=BLUE)
    c.path('M85 650 C260 630 230 500 440 450 S690 210 810 240',color='#344A9B')
    c.dot(440,450,fill='#344A9B');c.text(520,475,'全局交点',25)
    c.dot(335,470);c.text(285,510,'中心 c',24)
    c.line([(335,470),(595,525)]);c.text(650,548,'LOC0',24)
    c.line([(335,470),(405,330)]);c.text(440,310,'LOC1',24)
    c.line([(335,470),(250,185)],color='#9A7134');c.text(203,190,'法向 n',25)
    c.box(900,190,600,115,'① 平面求交\n得到 h(η, s∩)',BLUE)
    c.box(900,365,600,115,'② 局部变换\nRᵀ [h(η, s∩) − c]',BLUE)
    c.box(900,540,600,115,'③ 提取 LOC0 / 有效 LOC1\n测量减预测，再按方差标准化',BLUE)
    c.line([(1200,305),(1200,365)]);c.line([(1200,480),(1200,540)])
    c.box(180,735,1240,75,'求交有效性 ≠ 传感器有限边界检查；无效分量通过 mask 排除',GOLD,26)
    c.save('a10_intersection')


# --- Progressive three-stage diagrams -------------------------------------
# Trunk and inner-loop slots below are shared by seed and refit and are never
# recomputed per stage, so every stage renders each element at the same pixel
# position; only the muted set changes. That is what lets the three exports be
# shown back to back as one diagram growing rather than three separate figures.
T1 = (100, 145, 520, 90)
T2 = (100, 275, 520, 100)
T3 = (360, 475, 560, 150)
T4 = (100, 575, 520, 100)
T5 = (360, 745, 520, 110)
T6 = (95, 820, 430, 62)
TX = (560, 820, 300, 62)
I1 = (760, 145, 780, 90)
I2 = (760, 275, 780, 100)
I3 = (1150, 475, 720, 160)
I4 = (770, 590, 330, 95)
I5 = (1210, 590, 330, 95)
I6 = (1150, 768, 360, 110)
FOLD = (780, 390, 740, 170)


def _trunk(c, stage, texts, tone, refit=False):
    head, tail, exit_m = stage != 1, stage == 2, stage != 3
    c.box(*T1, texts['t1'], tone, 34, mute=head)
    c.box(*T2, texts['t2'], tone, 34, mute=head)
    c.line([(360,235),(360,275)], mute=head)
    c.diamond(*T3, texts['t3'], 30, mute=tail)
    if not (refit and stage == 2):
        c.line([(360,375),(360,400)], mute=head)
    c.box(*T4, texts['t4'], GOLD, 34, mute=tail)
    c.line([(360,550),(360,575)], '是', 322, 568, mute=tail)
    c.diamond(*T5, texts['t5'], 30, mute=tail)
    c.line([(360,675),(360,690)], mute=tail)
    c.box(*T6, texts['t6'], tone, 28, mute=tail)
    c.line([(360,800),(360,820)], '是', 322, 815, mute=tail)
    c.box(*TX, '失败退出并记录', RED, 26, mute=exit_m)
    c.line([(620,745),(710,745),(710,820)], '否', 668, 726, mute=exit_m)
    if refit:
        c.line([(80,475),(40,475),(40,325),(100,325)], '否', 63, 422, mute=tail)


def _fold(c, mute):
    c.box(*FOLD, 'LM 内循环\n求步 → 试算 → 接受 / 拒绝 → 调阻尼\n（第 2/3 页展开）',
          GREY, 28, mute=mute, dash=True)
    c.line([(640,475),(780,475)], '否', 710, 458, mute=mute)
    # Leave from the top edge so the return arrow never collides with the
    # "否" label on the entry arrow.
    c.line([(900,390),(900,250),(660,250),(660,325),(620,325)], '↺ 回 ②', 800, 236, mute=mute)
    c.line([(1150,560),(1150,625),(620,625)], '预算耗尽', 880, 606, mute=mute)


def _inner_common(c, tone, texts, refit=False):
    """Shared inner-loop geometry; the candidate test text differs per stage."""
    if refit:
        c.line([(620,325),(700,325),(700,190),(760,190)])
        c.text(684,262,'展开 ②',22,bold=True,anchor='end')
    else:
        c.line([(640,475),(700,475),(700,190),(760,190)], '否', 668, 455)
    c.box(*I1, texts['i1'], tone, 28)
    c.line([(1150,235),(1150,275)])
    c.box(*I2, texts['i2'], tone, 28)
    c.line([(1150,375),(1150,395)])
    c.diamond(*I3, texts['i3'], 28)
    c.box(*I4, '提交候选\n阻尼 ×0.1，有下限', tone, 26)
    c.box(*I5, '保留参数\n阻尼 ×10', RED, 26)
    # Both legs drop straight from the diamond tips: the far-right corridor is
    # reserved for the re-linearisation return edge, which would otherwise run
    # within 10 px of the "否" leg and read as a single line when projected.
    c.line([(790,475),(790,535),(935,535),(935,590)], '是', 828, 512)
    c.line([(1510,475),(1510,535),(1375,535),(1375,590)], '否', 1472, 512)
    c.line([(935,685),(935,705),(1150,705)], arrow=False)
    c.line([(1375,685),(1375,705),(1150,705)], arrow=False)
    c.line([(1150,705),(1150,713)])


def _stage_canvas(title, stage, conclusion):
    return Canvas(title, conclusion=conclusion, badge=f'{stage} / 3',
                  label_size=26, label_bold=True)


SEED_TEXTS = {
    't1': '① Helix 初值\n规范固定：去掉原点冗余',
    't2': '② 最近点 → 残差 → J, H, g\n逐径迹 decrement',
    't3': '③ 分块内全部径迹\n满足驻点判据？',
    't4': '④ 重算终态诊断\n驻点 · 非有限值 · 失效率',
    't5': '⑤ 主线检查通过？',
    't6': '恢复六参数 → refit',
    'i1': '未驻点径迹：求 LM 步，构造候选参数',
    'i2': '重新计算候选最近点\n目标函数及有效性',
    'i3': '逐径迹：有限且严格下降\n且不丢初始有效点？',
}
SEED_CONCLUSION = '通过点到 helix 的几何预拟合，为局部测量重拟合提供初值。'
SEED_TITLES = {1: 'SMT seed｜主干：一次预拟合的五步',
               2: 'SMT seed｜内循环：一步如何被接受或拒绝',
               3: 'SMT seed｜收敛与验收：何时停、何为通过'}

REFIT_TEXTS = {
    't1': '① 初始求交检查 → 规范固定\n初始检查失败则直接退出',
    't2': '② 完成一次 LM 更新\n计算候选与 change',
    't3': '③ 分块内最大 change\n< 10⁻¹⁰ 或预算耗尽？',
    't4': '④ 独立终态验收\n驻点 · 非有限值 · 失效率',
    't5': '⑤ 主线验收通过？',
    't6': '恢复六参数 → 固定径迹',
    'i1': '求交与残差 → J, H, g\n求 LM 步，构造候选参数',
    'i2': '重新求交\n计算候选目标函数',
    'i3': '逐径迹：候选目标\n严格下降？',
}
REFIT_CONCLUSION = '在固定工作几何上，以非目标层局部测量重新约束径迹参数。'
REFIT_TITLES = {1: 'Measurement refit｜主干：一次重拟合的五步',
                2: 'Measurement refit｜内循环：一步如何被接受或拒绝',
                3: 'Measurement refit｜收敛与验收：何时停、何为通过'}


def seed_stage(stage):
    c = _stage_canvas(SEED_TITLES[stage], stage, SEED_CONCLUSION)
    _trunk(c, stage, SEED_TEXTS, TEAL)
    if stage == 2:
        _inner_common(c, TEAL, SEED_TEXTS)
        c.diamond(*I6, '预算耗尽？', 28)
        # Budget exhaustion is the only way out of the loop other than
        # re-linearisation; both legs stay inside the right-hand block.
        c.line([(970,768),(700,768),(700,625),(620,625)], '是', 833, 750)
        c.line([(1330,768),(1555,768),(1555,120),(660,120),(660,325),(620,325)], '否 → 回 ②', 1368, 109)
    else:
        _fold(c, mute=stage == 3)
    c.save(f'p3_seed_s{stage}')


def refit_stage(stage):
    c = _stage_canvas(REFIT_TITLES[stage], stage, REFIT_CONCLUSION)
    _trunk(c, stage, REFIT_TEXTS, BLUE, refit=True)
    if stage == 2:
        _inner_common(c, BLUE, REFIT_TEXTS, refit=True)
        # refit has no per-track stationary freeze: the loop always falls
        # through to the block-wide change statistic and back to step ②.
        c.box(940, 713, 420, 95, '计算分块内最大 change', BLUE, 28)
        c.line([(940,760),(730,760),(730,475),(640,475)], '返回 ③', 825, 736)
    else:
        c.box(*FOLD, '② 的展开内容\n求交 → LM 候选 → 接受 / 拒绝\n计算 change（第 2/3 页展开）',
              GREY, 28, mute=stage == 3, dash=True)
        c.line([(620,325),(740,325),(740,430),(780,430)], arrow=False,
               dash=True, mute=stage == 3)
    c.save(f'p4_refit_s{stage}')


if __name__ == '__main__':
    seed();refit();gauge();closest();intersection();blocklm_band()
    for s in (1, 2, 3):
        seed_stage(s);refit_stage(s)
    print('Wrote 12 SVG diagrams to',OUT)
