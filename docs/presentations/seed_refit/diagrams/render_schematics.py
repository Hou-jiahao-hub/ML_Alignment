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


class Canvas:
    def __init__(self, title, subtitle='', width=1600, height=1000):
        self.w, self.h = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8 Z" fill="#53616D"/></marker></defs>',
                      '<rect width="100%" height="100%" fill="white"/>']
        self.text(65, 58, title, 34, bold=True, anchor='start')
        if subtitle:
            self.text(65, 98, subtitle, 22, anchor='start', color='#53616D')

    def text(self, x, y, lines, size=25, bold=False, anchor='middle', color=INK):
        if isinstance(lines, str):
            lines = lines.split('\n')
        for j, line in enumerate(lines):
            content = escape(line).replace('s∩', 's<tspan baseline-shift="super" font-size="70%">∩</tspan>')
            self.parts.append(f'<text x="{x}" y="{y+j*(size*1.45)}" font-family="Droid Sans Fallback, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{content}</text>')

    def box(self, x, y, w, h, lines, fill=TEAL, size=25):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#62777F" stroke-width="2"/>')
        n = len(lines.split('\n'))
        self.text(x+w/2, y+h/2-(n-1)*size*1.45/2+size*.32, lines, size)

    def diamond(self, cx, cy, w, h, lines, size=23):
        self.parts.append(f'<path d="M{cx-w/2} {cy} L{cx} {cy-h/2} L{cx+w/2} {cy} L{cx} {cy+h/2} Z" fill="{GOLD}" stroke="#9B8051" stroke-width="2"/>')
        n = len(lines.split('\n'))
        self.text(cx, cy-(n-1)*size*1.45/2+size*.32, lines, size)

    def line(self, points, label=None, lx=None, ly=None, arrow=True, color='#53616D', dash=False):
        d = 'M' + ' L'.join(f'{x},{y}' for x,y in points)
        dash_attr = 'stroke-dasharray="7 6"' if dash else ''
        arrow_attr = 'marker-end="url(#arrow)"' if arrow else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4" {dash_attr} {arrow_attr}/>')
        if label:
            self.text(lx, ly, label, 22, color=color)

    def path(self, d, color='#287A6D', width=4, fill='none', dash=False):
        dash_attr = 'stroke-dasharray="8 7"' if dash else ''
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" {dash_attr}/>')

    def dot(self, x, y, fill='#287A6D', r=7):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

    def save(self, name):
        (OUT / f'{name}.svg').write_text('\n'.join(self.parts)+'</svg>', encoding='utf-8')


def seed():
    c=Canvas('SMT seed｜几何预拟合', '非目标 SMT 点；五维优化；输出六个物理参数')
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
    c=Canvas('Measurement refit｜局部测量重拟合', '固定工作几何；非目标 measurement；终态验收后固定径迹')
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


if __name__ == '__main__':
    seed();refit();gauge();closest();intersection()
    print('Wrote 5 compact SVG diagrams to',OUT)
