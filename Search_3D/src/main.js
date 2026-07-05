import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.querySelector("#scene");

const state = {
  count: 100,
  mode: "search",
  algorithmId: "A01",
  scenario: "urban",
  running: true,
  elapsed: 0,
  selected: 0,
  showPaths: true,
  showCorridor: true,
  showVectors: true,
  showSafety: false,
  showGrid: false,
  showTrails: true,
  drones: [],
  obstacles: [],
  grid: null,
  distance: null,
  goals: [],
  conflicts: 0,
  averageSpacing: 0,
  frameMs: 16,
  lastStatsAt: 0,
};

const modeLabels = {
  search: "搜索 / 采样",
  optimize: "轨迹优化",
  avoid: "分布式避让",
  field: "反应式 / 场方法",
  learning: "学习类",
  central: "集中式协调",
};

const algorithmGroups = [
  {
    mode: "search",
    title: "1. 搜索 / 采样类全局规划",
    shortTitle: "搜索 / 采样",
    algorithms: [
      { id: "A01", name: "3D A*", summary: "栅格前端搜索，显示距离场与折线路径", profile: { speed: 1, lookahead: 1 } },
      { id: "A02", name: "Hybrid A*", summary: "带航向约束的搜索，路径转弯更克制", profile: { speed: 0.94, lookahead: 2, response: 2.7 } },
      { id: "A03", name: "JPS 3D", summary: "均匀栅格跳点剪枝，路径更少转折", profile: { speed: 1.08, lookahead: 2 } },
      { id: "A04", name: "RRT*", summary: "采样树风格，路径略带探索感", profile: { speed: 0.92, lookahead: 1, wander: 0.18 } },
      { id: "A05", name: "Informed RRT*", summary: "目标引导采样，收敛更集中", profile: { speed: 1.02, lookahead: 2, wander: 0.08 } },
      { id: "A06", name: "BIT*", summary: "批量采样搜索，强调全局代价层", profile: { speed: 0.98, lookahead: 2, response: 3.1 } },
      { id: "A07", name: "Kinodynamic A*", summary: "位置 + 速度状态搜索，动力学更保守", profile: { speed: 0.82, lookahead: 2, response: 2.25 } },
      { id: "A08", name: "Motion Primitives", summary: "Minimum-jerk primitive 前端", profile: { speed: 0.88, lookahead: 2, response: 2.45 } },
    ],
  },
  {
    mode: "optimize",
    title: "2. 轨迹优化后端",
    shortTitle: "轨迹优化",
    algorithms: [
      { id: "B01", name: "Minimum Snap", summary: "多项式平滑轨迹，四旋翼经典后端", profile: { speed: 1.08, lookahead: 3, response: 2.05 } },
      { id: "B02", name: "Minimum Jerk", summary: "最小加加速度，轨迹柔顺", profile: { speed: 0.98, lookahead: 3, response: 1.9 } },
      { id: "B03", name: "SFC + Convex QP", summary: "安全飞行走廊内做凸优化", profile: { speed: 1, lookahead: 2, corridor: 1.25 } },
      { id: "B04", name: "B-Spline + ESDF", summary: "梯度推离障碍，适合在线重规划", profile: { speed: 1.05, lookahead: 3, obstacleMargin: 3.35 } },
      { id: "B05", name: "Fast-Planner", summary: "ESDF 梯度 + B 样条快速优化", profile: { speed: 1.14, lookahead: 3, obstacleMargin: 3.15 } },
      { id: "B06", name: "GCOPTER / MINCO", summary: "稀疏参数化时空联合优化", profile: { speed: 1.22, lookahead: 4, response: 2.55 } },
    ],
  },
  {
    mode: "avoid",
    title: "3. 分布式机间避让",
    shortTitle: "分布式避让",
    algorithms: [
      {
        id: "C01",
        name: "3D ORCA",
        summary: "最优互反避障，速度空间半平面约束",
        profile: { speed: 1, avoidRange: 2.75, avoidWeight: 4.2, timeHorizon: 2.8, reciprocalShare: 0.52 },
      },
      { id: "C02", name: "RVO 3D", summary: "互惠速度障碍，局部避让直观", profile: { speed: 1.05, avoidRange: 2.35, avoidWeight: 3.7, timeHorizon: 2.4, reciprocalShare: 0.55 } },
      { id: "C03", name: "Buffered Voronoi Cells", summary: "每机限制在缓冲 Voronoi 胞内", profile: { speed: 0.92, avoidRange: 2.85, avoidWeight: 4.6, safetyScale: 1.12 } },
      { id: "C04", name: "DMPC", summary: "滚动优化并使用邻机预测轨迹", profile: { speed: 0.96, avoidRange: 2.7, avoidWeight: 4.35, response: 2.6 } },
      { id: "C05", name: "MADER / RMADER", summary: "异步去中心化轨迹 recheck", profile: { speed: 1.03, avoidRange: 2.65, avoidWeight: 4.7, startDelay: 0.06 } },
      { id: "C06", name: "EGO-Swarm", summary: "无 ESDF 梯度优化 + 广播轨迹互避", profile: { speed: 1.12, avoidRange: 2.55, avoidWeight: 4.1, obstacleMargin: 2.9 } },
      { id: "C07", name: "DCP / 错峰通过", summary: "时间维度冲突消解", profile: { speed: 0.9, avoidRange: 2.45, avoidWeight: 3.8, startDelay: 0.16 } },
      {
        id: "C08",
        name: "3D HRVO",
        summary: "混合互惠速度障碍，减轻振荡避让",
        profile: { speed: 1.02, avoidRange: 2.65, avoidWeight: 4.05, response: 3.2, timeHorizon: 2.6, reciprocalShare: 0.64, tangentBias: 0.22 },
      },
    ],
  },
  {
    mode: "field",
    title: "4. 反应式 / 场方法",
    shortTitle: "场方法",
    algorithms: [
      { id: "D01", name: "APF 3D", summary: "目标吸引 + 障碍排斥", profile: { speed: 0.94, obstacleMargin: 3.45, avoidWeight: 3.1 } },
      {
        id: "D02",
        name: "3D Boids 群体模型",
        summary: "分离、对齐、聚合三规则",
        profile: { speed: 0.9, avoidRange: 3.1, perceptionRadius: 6.2, separationWeight: 4.4, alignmentWeight: 0.58, cohesionWeight: 0.34 },
      },
      {
        id: "D03",
        name: "Olfati-Saber Flocking",
        summary: "带理论收敛性的 flocking",
        profile: { speed: 0.86, avoidRange: 3.25, perceptionRadius: 6.6, cohesionWeight: 0.44, alignmentWeight: 0.62, separationWeight: 4.2 },
      },
      {
        id: "D04",
        name: "Vasarhelyi Flocking",
        summary: "实机集群风格参数化 flocking",
        profile: { speed: 0.96, avoidRange: 3.0, perceptionRadius: 5.6, cohesionWeight: 0.33, alignmentWeight: 0.55, separationWeight: 4.0 },
      },
      {
        id: "D05",
        name: "3D 社会力模型",
        summary: "目标驱动力 + 个体排斥力 + 障碍排斥力",
        profile: { speed: 0.88, avoidRange: 4.6, avoidWeight: 3.55, obstacleMargin: 4.1, socialA: 4.4, socialB: 0.92, socialTau: 0.72 },
      },
    ],
  },
  {
    mode: "learning",
    title: "5. 学习类",
    shortTitle: "学习类",
    algorithms: [
      { id: "E01", name: "GLAS", summary: "GLAS 局部观测网络 + 注意力聚合 + 安全混合", profile: { speed: 1.05, avoidRange: 2.35, avoidWeight: 4.35 } },
      { id: "E02", name: "PRIMAL / PRIMAL2", summary: "PRIMAL 27 离散动作策略网络 + 局部占用评估", profile: { speed: 0.98, avoidRange: 2.25, avoidWeight: 4.1, wander: 0.12 } },
      { id: "E03", name: "Neural CBF", summary: "神经 CBF 屏障近似 + 速度 QP 安全滤波", profile: { speed: 1.0, avoidRange: 2.55, avoidWeight: 4.7, obstacleMargin: 3.2 } },
      { id: "E04", name: "RL + Safety Layer", summary: "RL 策略网络 + 短时域预测安全层", profile: { speed: 1.08, avoidRange: 2.4, avoidWeight: 4.55, wander: 0.1 } },
      { id: "E05", name: "End-to-End Swarm RL", summary: "端到端 CTDE MLP：局部观测直接输出速度修正", profile: { speed: 1.12, avoidRange: 2.2, avoidWeight: 4.0, wander: 0.2 } },
    ],
  },
  {
    mode: "central",
    title: "6. 集中式 / 编队与任务层",
    shortTitle: "集中式协调",
    algorithms: [
      { id: "F01", name: "MAPF 3D", summary: "集中式多智能体路径规划", profile: { speed: 0.92, laneCount: 5, startDelay: 0.08 } },
      { id: "F02", name: "CBS / ECBS", summary: "基于冲突的搜索与次优加速", profile: { speed: 0.9, laneCount: 7, startDelay: 0.11 } },
      { id: "F03", name: "PBS", summary: "优先级规划，适合编队调度", profile: { speed: 0.96, laneCount: 6, startDelay: 0.09 } },
      { id: "F04", name: "SCP 集中式轨迹", summary: "序列凸规划统一生成全体轨迹", profile: { speed: 0.84, laneCount: 8, response: 2.05, startDelay: 0.12 } },
      { id: "F05", name: "匈牙利 / 拍卖分配", summary: "目标分配 + 局部规划组合", profile: { speed: 1.02, laneCount: 5, startDelay: 0.07 } },
    ],
  },
];

const algorithms = algorithmGroups.flatMap((group) =>
  group.algorithms.map((algorithm) => ({
    ...algorithm,
    mode: group.mode,
    familyTitle: group.title,
    familyShortTitle: group.shortTitle,
  })),
);
const algorithmById = new Map(algorithms.map((algorithm) => [algorithm.id, algorithm]));

// ===== Algorithm learning cards: concept diagrams and descriptions =====

const DIAG = {
  grid: "rgba(255,255,255,0.09)",
  obstacle: "#4b555f",
  teal: "#2ee6d6",
  green: "#76e06f",
  amber: "#ffbf47",
  rose: "#ff667c",
  violet: "#b692ff",
  muted: "#a9b0b7",
};

function svgFrame(inner) {
  const markers = ["teal", "green", "amber", "rose", "violet", "muted"]
    .map(
      (key) =>
        `<marker id="arr-${key}" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="4.5" markerHeight="4.5" orient="auto"><path d="M0,0L8,4L0,8Z" fill="${DIAG[key]}"/></marker>`,
    )
    .join("");
  return `<svg viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg"><defs>${markers}</defs>${inner}</svg>`;
}

function dLine(x1, y1, x2, y2, color, width = 2, dash = "") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${DIAG[color]}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
}

function dArrow(x1, y1, x2, y2, color, width = 2, dash = "") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${DIAG[color]}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""} marker-end="url(#arr-${color})"/>`;
}

function dText(x, y, text, color = "muted", size = 10, anchor = "middle") {
  return `<text x="${x}" y="${y}" fill="${DIAG[color]}" font-size="${size}" text-anchor="${anchor}">${text}</text>`;
}

function dDot(x, y, r, color, opacity = 1) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${DIAG[color]}" opacity="${opacity}"/>`;
}

const diagCell = (cx, cy) => [20 + cx * 26 + 13, 15 + cy * 24 + 12];

function diagGridBase() {
  let g = "";
  for (let i = 0; i <= 10; i += 1) g += dLine(20 + i * 26, 15, 20 + i * 26, 135, "grid", 1);
  for (let j = 0; j <= 5; j += 1) g += dLine(20, 15 + j * 24, 280, 15 + j * 24, "grid", 1);
  for (const [cx, cy] of [[4, 1], [4, 2], [4, 3], [7, 3], [7, 4]]) {
    g += `<rect x="${20 + cx * 26 + 1}" y="${15 + cy * 24 + 1}" width="24" height="22" fill="${DIAG.obstacle}"/>`;
  }
  g += dDot(33, 123, 5, "green") + dDot(267, 27, 5, "rose");
  g += dText(33, 112, "起点", "green") + dText(267, 45, "目标", "rose");
  return g;
}

const diagPathCells = [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]];

function diagPathPolyline(color = "teal", width = 2.5) {
  const points = diagPathCells.map(([cx, cy]) => diagCell(cx, cy).join(",")).join(" ");
  return `<polyline points="${points}" fill="none" stroke="${DIAG[color]}" stroke-width="${width}" stroke-linejoin="round"/>`;
}

function diagramGridSearch(variant) {
  let extra = "";
  if (variant === "astar") {
    const wave = [[0, 3], [1, 4], [1, 2], [2, 3], [0, 2], [2, 4], [1, 1], [3, 2], [2, 1], [3, 3], [0, 1], [3, 0], [2, 0], [3, 4], [5, 1], [4, 4], [1, 0], [6, 1], [5, 2]];
    wave.forEach(([cx, cy], i) => {
      extra += `<rect x="${20 + cx * 26 + 2}" y="${15 + cy * 24 + 2}" width="22" height="20" fill="${DIAG.teal}" opacity="${Math.max(0.05, 0.24 - i * 0.011)}"/>`;
    });
    extra += diagPathPolyline() + dText(150, 148, "开放/关闭列表按 f = g + h 波前扩展", "muted");
  } else if (variant === "hybrid") {
    extra += `<path d="M33,123 C 90,120 80,60 130,38 C 175,20 220,24 267,27" fill="none" stroke="${DIAG.teal}" stroke-width="2.5"/>`;
    extra += dArrow(78, 106, 96, 92, "amber") + dArrow(150, 32, 170, 27, "amber");
    extra += dText(150, 148, "航向进入状态 · 每步限转 · 转向有代价", "muted");
  } else if (variant === "jps") {
    const jumps = [[0, 4], [3, 1], [4, 0], [9, 0]];
    const pts = jumps.map(([cx, cy]) => diagCell(cx, cy));
    for (let i = 0; i < pts.length - 1; i += 1) extra += dLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], "teal", 2.5);
    extra += pts.slice(1, -1).map(([x, y]) => dDot(x, y, 4.5, "amber")).join("");
    extra += dText(118, 55, "跳点", "amber") + dText(150, 148, "沿直线跳跃剪枝，仅在 forced neighbor 生成节点", "muted");
  } else if (variant === "kino") {
    extra += diagPathPolyline();
    const arrowsAt = [[1, 3], [3, 1], [5, 0], [7, 0]];
    for (const [cx, cy] of arrowsAt) {
      const [x, y] = diagCell(cx, cy);
      extra += dArrow(x, y, x + 16, y - (cy > 0 ? 13 : 0), "violet", 1.8);
    }
    extra += dText(150, 148, "状态 = 位置 × 速度方向，扩展受加速度约束", "muted");
  } else if (variant === "primitives") {
    extra += `<path d="M33,123 Q 59,115 72,99 Q 86,81 98,63 Q 110,44 124,32 Q 140,22 163,25 Q 200,29 228,27 Q 248,26 267,27" fill="none" stroke="${DIAG.teal}" stroke-width="2.5"/>`;
    for (const [x, y] of [[72, 99], [98, 63], [124, 32], [163, 25], [228, 27]]) extra += dDot(x, y, 3.6, "violet");
    extra += dText(150, 148, "预置可行轨迹段（基元）逐段拼接搜索", "muted");
  }
  return svgFrame(diagGridBase() + extra);
}

function diagramTree(variant) {
  const nodes = [
    [33, 123], [62, 100], [80, 126], [55, 66], [96, 84], [120, 108], [104, 52], [140, 76],
    [150, 118], [168, 46], [186, 88], [206, 62], [226, 96], [238, 40], [256, 70], [186, 124],
  ];
  const edges = [[0, 1], [0, 2], [1, 3], [1, 4], [4, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [9, 11], [10, 12], [11, 13], [12, 14], [8, 15]];
  const best = [0, 1, 4, 7, 10, 12, 14];
  let g = "";
  if (variant === "informed") {
    g += `<ellipse cx="150" cy="75" rx="128" ry="42" transform="rotate(-17 150 75)" fill="${DIAG.violet}" opacity="0.08" stroke="${DIAG.violet}" stroke-dasharray="5 4"/>`;
    g += dText(228, 118, "informed 椭球采样域", "violet");
  }
  if (variant === "batches") {
    g += `<ellipse cx="150" cy="75" rx="132" ry="52" transform="rotate(-16 150 75)" fill="none" stroke="${DIAG.muted}" stroke-dasharray="5 4" opacity="0.6"/>`;
    g += `<ellipse cx="152" cy="72" rx="112" ry="34" transform="rotate(-16 150 75)" fill="${DIAG.violet}" opacity="0.08" stroke="${DIAG.violet}" stroke-dasharray="5 4"/>`;
    g += dText(238, 126, "批次 1 → 批次 2 收缩", "violet");
  }
  for (const [a, b] of edges) {
    const onBest = best.includes(a) && best.includes(b) && Math.abs(best.indexOf(a) - best.indexOf(b)) === 1;
    g += dLine(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], onBest ? "teal" : "muted", onBest ? 2.5 : 1.2);
  }
  for (let i = 0; i < nodes.length; i += 1) {
    g += dDot(nodes[i][0], nodes[i][1], i === 0 ? 5 : 2.6, i === 0 ? "green" : best.includes(i) ? "teal" : "muted");
  }
  g += dLine(256, 70, 267, 47, "teal", 2.5) + dDot(267, 40, 5, "rose");
  g += dText(33, 112, "起点", "green") + dText(267, 58, "目标", "rose");
  if (!variant) g += dText(150, 145, "随机采样生长树 · 邻域内选优父节点并 rewire", "muted");
  else g += dText(150, 145, variant === "informed" ? "有解后仅在椭球内采样，加速收敛" : "批量采样 + 图搜索，边碰撞惰性检测", "muted");
  return svgFrame(g);
}

function diagramTrajectory(variant) {
  const rawPts = "30,118 70,96 96,110 128,66 160,84 196,40 226,56 268,32";
  let g = `<polyline points="${rawPts}" fill="none" stroke="${DIAG.amber}" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.85"/>`;
  const smooth = `M30,118 C 70,104 100,96 128,74 C 156,54 196,58 226,46 C 244,40 256,36 268,32`;
  if (variant === "speed") {
    g += `<path d="M30,118 C 70,104 100,96 128,74" fill="none" stroke="${DIAG.teal}" stroke-width="3"/>`;
    g += `<path d="M128,74 C 156,54 178,56 196,52" fill="none" stroke="${DIAG.amber}" stroke-width="3"/>`;
    g += `<path d="M196,52 C 220,48 250,38 268,32" fill="none" stroke="${DIAG.teal}" stroke-width="3"/>`;
    g += dText(163, 40, "急弯降速", "amber");
  } else {
    g += `<path d="${smooth}" fill="none" stroke="${DIAG.teal}" stroke-width="2.6"/>`;
  }
  if (variant === "corridor") {
    for (const [x, y, r] of [[30, 118, 0], [80, 98, -18], [130, 72, -24], [186, 54, -12], [240, 40, -10]]) {
      g += `<rect x="${x - 6}" y="${y - 15}" width="64" height="30" rx="6" transform="rotate(${r} ${x} ${y})" fill="${DIAG.green}" opacity="0.1" stroke="${DIAG.green}" stroke-opacity="0.5"/>`;
    }
    g += dText(150, 145, "轨迹被约束在重叠凸走廊内做平滑", "muted");
  } else if (variant === "ctrl") {
    const ctrl = [[30, 118], [84, 92], [132, 84], [176, 46], [222, 52], [268, 32]];
    g += `<polyline points="${ctrl.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${DIAG.violet}" stroke-width="1.2" stroke-dasharray="3 3"/>`;
    g += ctrl.map(([x, y]) => dDot(x, y, 3.6, "violet")).join("");
    g += dArrow(132, 84, 118, 104, "rose", 1.8) + dText(112, 118, "ESDF 梯度推离", "rose");
    g += dText(150, 145, "优化 B 样条控制点：平滑项 + 距离场斥力", "muted");
  } else if (variant === "sparse") {
    const wp = [[30, 118], [128, 74], [226, 46], [268, 32]];
    g += wp.map(([x, y]) => dDot(x, y, 4.4, "violet")).join("");
    g += dText(150, 145, "只优化少量路标，段内多项式解析最优", "muted");
  } else if (variant === "speed") {
    g += dText(150, 145, "B 样条 + ESDF 优化，按曲率重分配速度", "muted");
  } else {
    g += dText(150, 145, "最小化高阶导数：折线 → 控制代价最小的平滑轨迹", "muted");
  }
  g += dDot(30, 118, 4.5, "green") + dDot(268, 32, 4.5, "rose");
  g += dText(76, 84, "初始折线", "amber");
  return svgFrame(g);
}

function diagramVelocityObstacle(variant) {
  let g = dDot(48, 116, 7, "teal") + dText(48, 136, "本机 A", "teal");
  g += dDot(112, 38, 7, "muted") + dText(112, 26, "邻机 B", "muted");
  g += dArrow(112, 38, 138, 58, "muted", 1.6);
  const ax = 178;
  const ay = 108;
  g += `<path d="M${ax},${ay} L 268,34 L 292,86 Z" fill="${DIAG.rose}" opacity="0.13" stroke="${DIAG.rose}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
  g += dText(258, 66, "碰撞速度集", "rose", 10);
  g += dArrow(ax, ay, 248, 62, "amber", 2) + dText(232, 92, "期望速度", "amber");
  if (variant === "orca") {
    g += dLine(196, 44, 292, 108, "teal", 1.6, "5 4");
    g += dArrow(ax, ay, 224, 116, "teal", 2.2) + dText(224, 132, "半平面内取最近速度", "teal");
    g += dText(ax - 4, ay + 16, "各承担一半修正", "muted");
  } else {
    g += dArrow(ax, ay, 216, 120, "teal", 2.2) + dText(226, 134, "修正到锥外", "teal");
    const apexLabel = variant === "rvo" ? "锥顶 = (vA+vB)/2" : variant === "hrvo" ? "混合锥顶（偏置选边）" : "锥顶 = vB";
    g += dText(ax - 8, ay + 16, apexLabel, "violet");
  }
  return svgFrame(g);
}

function diagramVoronoiCell() {
  const hex = "150,26 224,52 232,104 158,132 84,110 72,54";
  const inner = "150,42 206,62 212,98 156,118 100,102 92,62";
  let g = `<polygon points="${hex}" fill="none" stroke="${DIAG.muted}" stroke-width="1.4"/>`;
  g += `<polygon points="${inner}" fill="${DIAG.teal}" opacity="0.08" stroke="${DIAG.teal}" stroke-dasharray="5 4"/>`;
  g += dDot(150, 82, 6, "teal");
  for (const [x, y] of [[36, 30], [258, 28], [278, 96], [40, 124], [180, 12]]) g += dDot(x, y, 4.5, "muted");
  g += dArrow(150, 82, 192, 96, "teal", 2.2);
  g += dText(66, 26, "邻机", "muted") + dText(196, 40, "Voronoi 胞", "muted") + dText(152, 132, "缓冲收缩后的安全胞", "teal");
  g += dText(150, 148, "每步速度被约束：下一位置不出自己的胞", "muted");
  return svgFrame(g);
}

function diagramRollout() {
  let g = dDot(42, 78, 7, "teal");
  const fans = [
    ["M42,78 C 90,20 160,14 236,20", "muted"],
    ["M42,78 C 96,44 170,40 244,44", "muted"],
    ["M42,78 C 100,72 180,72 252,74", "teal"],
    ["M42,78 C 96,108 170,112 244,108", "muted"],
    ["M42,78 C 90,132 160,138 232,134", "muted"],
  ];
  for (const [d, color] of fans) {
    g += `<path d="${d}" fill="none" stroke="${DIAG[color]}" stroke-width="${color === "teal" ? 2.6 : 1.3}"${color === "teal" ? "" : ' stroke-dasharray="4 3"'}/>`;
  }
  g += dDot(224, 30, 5, "muted") + dArrow(224, 30, 168, 20, "rose", 1.6, "4 3");
  g += dText(236, 16, "邻机预测轨迹", "rose");
  g += `<text x="176" y="30" fill="${DIAG.rose}" font-size="14" text-anchor="middle">✕</text>`;
  g += dText(252, 90, "代价最低", "teal");
  g += dText(150, 148, "候选控制 × N 步 rollout，滚动执行第一步", "muted");
  return svgFrame(g);
}

function diagramCommit() {
  let g = `<path d="M28,110 C 90,96 180,84 272,74" fill="none" stroke="${DIAG.teal}" stroke-width="2.6"/>`;
  g += dDot(28, 110, 6, "teal") + dText(28, 128, "本机", "teal");
  g += `<path d="M212,16 C 196,52 178,84 168,120" fill="none" stroke="${DIAG.amber}" stroke-width="1.6" stroke-dasharray="5 4"/>`;
  g += dText(238, 16, "邻机承诺轨迹", "amber");
  g += `<circle cx="182" cy="80" r="14" fill="none" stroke="${DIAG.rose}" stroke-dasharray="3 3"/>`;
  g += `<text x="182" y="85" fill="${DIAG.rose}" font-size="12" text-anchor="middle">✕</text>`;
  g += `<path d="M120,94 C 150,120 200,124 262,108" fill="none" stroke="${DIAG.green}" stroke-width="2.2"/>`;
  g += `<text x="262" y="126" fill="${DIAG.green}" font-size="12" text-anchor="middle">✓ 重新承诺</text>`;
  g += dText(150, 148, "check 通过才承诺；recheck 失败则回退 / 减速", "muted");
  return svgFrame(g);
}

function diagramDetour() {
  let g = dLine(30, 80, 270, 80, "muted", 1.4, "5 4");
  g += dDot(30, 80, 6, "teal");
  g += dArrow(150, 18, 150, 66, "amber", 1.6, "4 3") + dText(150, 12, "邻机广播轨迹", "amber");
  g += dDot(150, 80, 5, "rose") + dText(178, 96, "预测冲突点", "rose");
  g += `<path d="M30,80 C 80,82 108,116 150,116 C 192,116 220,84 270,80" fill="none" stroke="${DIAG.teal}" stroke-width="2.6"/>`;
  g += dDot(270, 80, 5, "rose");
  g += dText(150, 145, "沿切向平滑绕行，不依赖 ESDF 全场", "muted");
  return svgFrame(g);
}

function diagramStagger() {
  let g = `<rect x="138" y="14" width="24" height="46" fill="${DIAG.obstacle}"/><rect x="138" y="94" width="24" height="44" fill="${DIAG.obstacle}"/>`;
  const rows = [[40, "t + 0", "teal"], [76, "t + Δ", "amber"], [112, "t + 2Δ", "violet"]];
  rows.forEach(([y, t, color], i) => {
    g += dDot(34, y, 5.5, color);
    g += dArrow(46, y, 128 - i * 4, 77 - (77 - y) * 0.25, color, 1.8);
    g += dText(34, y - 12, t, color);
  });
  g += dArrow(166, 77, 268, 77, "green", 2.2) + dText(224, 66, "依次通过瓶颈", "green");
  g += dText(150, 148, "空间路径不变，把冲突消解移到时间维", "muted");
  return svgFrame(g);
}

function diagramForces(exponential) {
  let g = `<rect x="96" y="96" width="52" height="30" fill="${DIAG.obstacle}"/>`;
  if (exponential) {
    for (let r = 1; r <= 3; r += 1) {
      g += `<rect x="${96 - r * 11}" y="${96 - r * 11}" width="${52 + r * 22}" height="${30 + r * 22}" rx="${r * 6}" fill="none" stroke="${DIAG.rose}" opacity="${0.4 - r * 0.11}"/>`;
    }
  }
  g += dDot(160, 66, 7, "teal") + dDot(268, 56, 6, "rose") + dText(268, 42, "目标", "rose");
  g += dDot(120, 30, 5, "muted") + dText(104, 20, "邻机", "muted");
  g += dArrow(160, 66, 226, 60, "green", 2.2) + dText(196, 48, "吸引", "green");
  g += dArrow(160, 66, 186, 34, "rose", 1.8) + dArrow(148, 78, 172, 100, "rose", 0.01);
  g += dArrow(160, 66, 184, 96, "rose", 1.8);
  g += dText(206, 106, exponential ? "指数斥力 A·e^((r−d)/B)" : "距离越近斥力越强", "rose");
  g += dArrow(160, 66, 236, 84, "teal", 2.4) + dText(244, 100, "合力", "teal");
  g += dText(150, 148, exponential ? "驱动力 (v*−v)/τ + 个体/障碍指数斥力" : "沿吸引 + 排斥势场的负梯度运动", "muted");
  return svgFrame(g);
}

function diagramBoids() {
  let g = "";
  const panel = (cx, title, inner) => dText(cx, 138, title, "teal", 11) + inner;
  let sep = "";
  for (const [x, y, ex, ey] of [[52, 62, 30, 42], [70, 78, 92, 92], [58, 92, 38, 112]]) {
    sep += dDot(x, y, 4, "teal") + dArrow(x, y, ex, ey, "rose", 1.6);
  }
  g += panel(60, "分离", sep);
  let ali = "";
  for (const [x, y] of [[136, 60], [158, 76], [140, 96]]) {
    ali += dDot(x, y, 4, "teal") + dArrow(x, y, x + 30, y - 8, "green", 1.6);
  }
  g += panel(150, "对齐", ali);
  let coh = "";
  const center = [244, 78];
  for (const [x, y] of [[220, 54], [268, 60], [238, 106]]) {
    coh += dDot(x, y, 4, "teal") + dArrow(x, y, center[0] + (x < center[0] ? -6 : 6), center[1], "amber", 1.6);
  }
  coh += `<text x="${center[0]}" y="${center[1] + 4}" fill="${DIAG.amber}" font-size="11" text-anchor="middle">×</text>`;
  g += panel(244, "聚合", coh);
  g += dLine(105, 30, 105, 120, "grid", 1) + dLine(195, 30, 195, 120, "grid", 1);
  g += dText(150, 22, "三条局部规则 → 群体行为涌现", "muted");
  return svgFrame(g);
}

function diagramLattice() {
  const center = [150, 72];
  const ring = [];
  for (let i = 0; i < 6; i += 1) {
    const angle = (Math.PI / 3) * i + 0.28;
    ring.push([center[0] + Math.cos(angle) * 46, center[1] + Math.sin(angle) * 40]);
  }
  let g = "";
  for (const [x, y] of ring) g += dLine(center[0], center[1], x, y, "teal", 1.2);
  for (let i = 0; i < 6; i += 1) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % 6];
    g += dLine(x1, y1, x2, y2, "teal", 1.2);
  }
  g += dDot(center[0], center[1], 5, "teal");
  for (const [x, y] of ring) g += dDot(x, y, 4.4, "teal") + dArrow(x, y, x + 20, y - 6, "green", 1.4);
  g += dText(178, 52, "d", "amber", 12);
  g += dText(150, 138, "φ 作用函数使间距收敛到 d（α-lattice）+ 速度一致", "muted");
  return svgFrame(g);
}

function diagramBraking() {
  let g = dArrow(46, 124, 282, 124, "muted", 1.4) + dArrow(46, 124, 46, 18, "muted", 1.4);
  g += dText(270, 138, "机间距离 r", "muted") + dText(28, 30, "Δv", "muted");
  g += `<path d="M60,122 Q 120,118 170,88 T 276,30" fill="none" stroke="${DIAG.teal}" stroke-width="2.4"/>`;
  g += `<path d="M60,122 Q 120,118 170,88 T 276,30 L 276,18 L 60,18 Z" fill="${DIAG.rose}" opacity="0.07"/>`;
  g += dDot(190, 46, 4.5, "rose") + dArrow(190, 46, 190, 72, "rose", 1.8);
  g += dText(232, 44, "超出 → 摩擦力拉齐", "rose");
  g += dText(150, 108, "允许速度差 D(r)", "teal");
  g += dText(150, 148, "距离越近允许速度差越小，符合理想制动", "muted");
  return svgFrame(g);
}

function diagramPolicy(variant) {
  const box = (x, y, w, h, title, color = "muted") =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="rgba(255,255,255,0.05)" stroke="${DIAG[color]}"/>` +
    dText(x + w / 2, y + h / 2 + 4, title, color === "muted" ? "muted" : color, 11);
  let g = box(14, 56, 66, 38, "局部观测");
  g += dArrow(80, 75, 100, 75, "muted", 1.6);
  if (variant === "mlp") {
    g += `<rect x="102" y="42" width="84" height="66" rx="7" fill="rgba(255,255,255,0.05)" stroke="${DIAG.violet}"/>`;
    const layers = [[116, [56, 74, 92]], [144, [50, 66, 82, 98]], [172, [62, 86]]];
    for (let l = 0; l < layers.length - 1; l += 1) {
      for (const y1 of layers[l][1]) for (const y2 of layers[l + 1][1]) g += dLine(layers[l][0], y1, layers[l + 1][0], y2, "violet", 0.5);
    }
    for (const [x, ys] of layers) for (const y of ys) g += dDot(x, y, 3, "violet");
    g += dText(144, 122, "端到端策略网络", "violet");
    g += dArrow(186, 75, 216, 75, "muted", 1.6);
    g += dArrow(226, 75, 278, 75, "teal", 2.6) + dText(252, 62, "动作", "teal");
  } else if (variant === "discrete") {
    g += box(102, 42, 78, 66, "", "violet") + dText(141, 60, "策略打分", "violet", 11);
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) {
        const on = r === 0 && c === 2;
        g += `<rect x="${118 + c * 16}" y="${68 + r * 12}" width="14" height="10" fill="${on ? DIAG.teal : "rgba(255,255,255,0.09)"}"/>`;
      }
    }
    g += dArrow(180, 75, 216, 75, "muted", 1.6);
    g += dArrow(226, 75, 278, 55, "teal", 2.6) + dText(248, 42, "离散动作", "teal");
  } else if (variant === "blend") {
    g += box(102, 30, 78, 34, "策略 π", "violet");
    g += box(102, 86, 78, 34, "安全动作", "amber");
    g += dArrow(180, 47, 216, 68, "violet", 1.8) + dArrow(180, 103, 216, 82, "amber", 1.8);
    g += `<circle cx="226" cy="75" r="12" fill="rgba(255,255,255,0.05)" stroke="${DIAG.teal}"/>` + dText(226, 79, "λ", "teal", 12);
    g += dArrow(238, 75, 284, 75, "teal", 2.6);
    g += dText(150, 145, "越接近危险 λ 越大，越偏向安全动作", "muted");
    return svgFrame(g);
  } else {
    g += box(102, 56, 70, 38, "策略 π", "violet");
    g += dArrow(172, 75, 192, 75, "muted", 1.6);
    g += box(194, 56, 72, 38, variant === "filter" ? "CBF 滤波" : "安全层", "amber");
    g += dArrow(266, 75, 292, 75, "teal", 2.6);
    g += dText(150, 130, variant === "filter" ? "约束 ḣ ≥ −αh：最小改动策略输出以保安全" : "预测将碰撞时修正 / 抑制策略输出", "muted");
    return svgFrame(g);
  }
  g += dText(150, 145, variant === "mlp" ? "观测 → 网络前向 → 控制，行为由奖励涌现" : "局部观测选离散动作，让路行为由训练涌现", "muted");
  return svgFrame(g);
}

function diagramGantt(variant) {
  const rows = [
    [34, 96, "teal"],
    [62, 128, "amber"],
    [90, 74, "violet"],
    [118, 108, "green"],
  ];
  const starts = variant === "conflict" ? [30, 30, 30, 30] : [30, 58, 88, 122];
  let g = dArrow(24, 134, 286, 134, "muted", 1.4) + dText(270, 148, "时间", "muted");
  rows.forEach(([y, w, color], i) => {
    g += `<rect x="${starts[i]}" y="${y - 8}" width="${w}" height="16" rx="5" fill="${DIAG[color]}" opacity="0.55"/>`;
    if (variant === "priority") g += dText(16, y + 4, String(i + 1), "amber", 11);
    else g += dText(16, y + 4, `#${i + 1}`, "muted", 9);
  });
  if (variant === "conflict") {
    g += `<rect x="66" y="24" width="20" height="48" fill="${DIAG.rose}" opacity="0.22" stroke="${DIAG.rose}" stroke-dasharray="3 3"/>`;
    g += dText(76, 16, "冲突", "rose");
    g += dArrow(96, 48, 150, 40, "rose", 1.4, "3 3") + dText(196, 36, "加约束分支重规划", "rose");
    g += `<rect x="152" y="54" width="128" height="16" rx="5" fill="${DIAG.amber}" opacity="0.55"/>`;
    g += dText(150, 148, "两层搜索：底层 A* + 顶层冲突树", "muted");
  } else if (variant === "priority") {
    g += dText(150, 148, "按优先级顺序规划，低优先级避让高优先级", "muted");
  } else {
    g += dText(150, 148, "体素 × 时隙保留表：延迟起飞直到全程无冲突", "muted");
  }
  return svgFrame(g);
}

function diagramJoint() {
  let g = `<path d="M24,44 C 100,44 150,96 276,80" fill="none" stroke="${DIAG.muted}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
  g += `<path d="M24,76 C 110,76 160,70 276,58" fill="none" stroke="${DIAG.muted}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
  g += `<path d="M24,108 C 100,108 150,60 276,36" fill="none" stroke="${DIAG.muted}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
  g += `<path d="M24,44 C 100,40 170,34 276,30" fill="none" stroke="${DIAG.teal}" stroke-width="2.2"/>`;
  g += `<path d="M24,76 C 110,74 170,66 276,60" fill="none" stroke="${DIAG.green}" stroke-width="2.2"/>`;
  g += `<path d="M24,108 C 100,108 170,98 276,92" fill="none" stroke="${DIAG.violet}" stroke-width="2.2"/>`;
  g += dArrow(150, 72, 150, 48, "rose", 1.6) + dArrow(150, 78, 150, 96, "rose", 1.6);
  g += dText(186, 118, "迭代推挤 + 平滑（凸化的分离约束）", "muted");
  g += dText(150, 145, "集中式联合优化全体轨迹，直至无冲突", "muted");
  return svgFrame(g);
}

function diagramAssign() {
  const drones = [[64, 34], [64, 66], [64, 98], [64, 130]];
  const goals = [[240, 34], [240, 66], [240, 98], [240, 130]];
  const match = [[0, 1], [1, 0], [2, 3], [3, 2]];
  let g = "";
  g += dLine(64, 34, 240, 98, "muted", 1, "4 3");
  g += `<text x="150" y="60" fill="${DIAG.rose}" font-size="13" text-anchor="middle">✕</text>`;
  for (const [a, b] of match) g += dLine(drones[a][0] + 8, drones[a][1], goals[b][0] - 8, goals[b][1], "teal", 1.8);
  for (const [x, y] of drones) g += dDot(x, y, 5.5, "teal");
  for (const [x, y] of goals) g += dDot(x, y, 5.5, "rose");
  g += dText(64, 18, "无人机", "teal") + dText(240, 18, "目标 (价格 p)", "rose");
  g += dText(150, 148, "二分图最小代价匹配：竞价-涨价迭代收敛", "muted");
  return svgFrame(g);
}

const algorithmDetails = {
  A01: {
    status: "已实现",
    diagram: () => diagramGridSearch("astar"),
    principle:
      "在三维体素栅格上做最优图搜索：维护开放列表，每次弹出 f = g + h 最小的节点扩展，g 是起点到该节点的累计代价，h 是到目标的启发式下界。只要 h 不高估真实代价（可采纳），找到的路径就是最优的。",
    traits: ["完备且最优，是栅格规划的基准算法", "26 邻域扩展，代价与内存随栅格规模线性增长", "适合静态已知环境，动态环境需要重规划"],
    demo: "每个起点体素运行一次严格的 open/closed A*，路径经视线简化与样条平滑后跟踪。",
  },
  A02: {
    status: "已实现",
    diagram: () => diagramGridSearch("hybrid"),
    principle:
      "把航向角加入搜索状态 (x, y, z, θ)：扩展时只允许与当前航向相近的运动方向，并对转向加代价，搜出的路径天然满足转弯约束。原版（斯坦福 Junior）用连续曲率弧线扩展并配合解析扩展命中目标。",
    traits: ["路径平滑少急转，适合有最小转弯半径的载具", "状态空间是 A* 的数倍（乘以航向离散数）", "最优性受航向离散化影响"],
    demo: "16 航向 Hybrid A*：连续弧段运动基元、爬升动作、曲率/转向代价与弧段碰撞检查。",
  },
  A03: {
    status: "已实现",
    diagram: () => diagramGridSearch("jps"),
    principle:
      "均匀代价栅格上 A* 的无损加速：利用路径对称性沿直线“跳跃”，跳过中间节点，只在遇到 forced neighbor（旁边被障碍堵住又重新打开的格子）或目标时才生成节点，堆操作减少一个数量级。",
    traits: ["与 A* 同样最优（均匀代价网格）", "扩展节点数远少于 A*，速度快数倍到数十倍", "3D forced-neighbor 与 no-corner-cut 规则实现复杂"],
    demo: "完整 3D JPS：方向剪枝、自然邻居、forced-neighbor、递归投影跳跃与 no-corner-cut 检查。",
  },
  A04: {
    status: "已实现",
    diagram: () => diagramTree(),
    principle:
      "反复随机采样并把采样点连向树上最近的节点（限制单步长度）生长出搜索树；RRT* 在此之上于新节点邻域内重新选择代价最低的父节点，并把邻居 rewire 到更优路径上，使解随迭代渐进最优。",
    traits: ["不需要栅格化，适合高维与复杂约束空间", "任意时间算法：随时可取当前最优解，越算越好", "解带随机性，通常需要后处理平滑"],
    demo: "完整 RRT*：动态近邻半径、最优父节点、rewire 代价传播、分支剪枝与目标连接。",
  },
  A05: {
    status: "已实现",
    diagram: () => diagramTree("informed"),
    principle:
      "先按 RRT* 找到初始解；之后把采样域收缩到以起点和终点为焦点、长轴等于当前最优代价的椭球内——椭球外的任何点都不可能改进当前解，因此收敛速度大幅提升。",
    traits: ["首解之后的收敛速度显著快于 RRT*", "椭球随解变优持续收缩，聚焦搜索", "找到首解之前与普通 RRT* 相同"],
    demo: "RRT* 首解后进入 prolate hyperspheroid informed 采样，并用当前最优代价做分支剪枝。",
  },
  A06: {
    status: "已实现",
    diagram: () => diagramTree("batches"),
    principle:
      "按“批”生成 informed 采样点，把这批点视作隐式图，用类 A* 的最佳优先方式按估计总代价处理边，碰撞检测按需惰性执行；每批结束用当前最优解收缩下一批采样域，兼得图搜索的有序与采样的可扩展。",
    traits: ["边按潜在价值排序处理，无谓碰撞检测少", "任意时间且渐进最优", "显式边队列与批次剪枝，适合高维采样图"],
    demo: "BIT* 批量 informed 采样：顶点队列扩展候选边，边队列按 g+c+h 排序，惰性碰撞检测并逐批剪枝。",
  },
  A07: {
    status: "已实现",
    diagram: () => diagramGridSearch("kino"),
    principle:
      "把速度纳入搜索状态，扩展时施加加速度约束——新的运动方向必须与当前速度方向足够接近，剧烈变向被禁止或付出高代价，因此搜出的轨迹动力学连续、可直接飞行。",
    traits: ["输出动力学可行轨迹，无需事后修正", "状态维度高，搜索空间明显变大", "分辨率与实时性需要权衡"],
    demo: "(体素 × 速度) Kinodynamic A*：离散速度状态、27 加速度动作、加速度/jerk 代价与动力学可达扩展。",
  },
  A08: {
    status: "已实现",
    diagram: () => diagramGridSearch("primitives"),
    principle:
      "先离线构造一小库“运动基元”——固定时长、满足动力学（如最小 jerk）的短轨迹段；在线搜索时逐段拼接基元并对每段做碰撞检测。基元库保证了任何拼接结果都可行。",
    traits: ["轨迹质量与机动风格由基元库决定", "搜索分支小、在线速度快", "表达能力受基元库限制"],
    demo: "运动基元库搜索：二阶速度拼接弧段、连续曲线采样碰撞检查、曲率代价与基元级 A*。",
  },
  B01: {
    status: "已实现",
    diagram: () => diagramTrajectory(),
    principle:
      "四旋翼的微分平坦性使位置轨迹的四阶导数 (snap) 直接对应控制输入。Minimum Snap（Mellinger & Kumar 2011）在航点约束下最小化 ∫‖snap‖²，解一个多项式系数的二次规划，得到控制代价最小的平滑轨迹。",
    traits: ["四旋翼轨迹规划的经典基线", "QP / 闭式求解，效率高", "本身不处理障碍，需配合走廊或重规划"],
    demo: "闭式分段 7 次 minimum snap 多项式：航点位置/速度/加速度/jerk 约束 + ESDF 净空投影。",
  },
  B02: {
    status: "已实现",
    diagram: () => diagramTrajectory(),
    principle:
      "同一框架下把目标函数换成三阶导数 (jerk)：jerk 与机体角速度相关，最小化它得到姿态变化柔和的轨迹，求解比 minimum snap 更轻，平滑度略低。",
    traits: ["姿态变化柔和，乘性噪声小", "阶数低于 snap，数值条件更好", "同样需要外部处理障碍"],
    demo: "闭式分段 5 次 minimum jerk 多项式：航点位置/速度/加速度约束 + ESDF 净空投影。",
  },
  B03: {
    status: "已实现",
    diagram: () => diagramTrajectory("corridor"),
    principle:
      "先沿初始路径把自由空间分解成一串互相重叠的凸多面体（安全飞行走廊 SFC），再以“轨迹各段必须留在对应凸体内”为约束做凸优化平滑。碰撞约束被凸化后，求解可靠且有全局最优保证。",
    traits: ["把非凸避障问题变成凸问题，求解稳定", "走廊质量直接决定轨迹质量", "实机系统（如 EGO、Faster）广泛使用"],
    demo: "安全飞行走廊 QP：重叠 AABB 凸走廊约束 + 平滑/路径跟踪目标 + ESDF 投影。",
  },
  B04: {
    status: "已实现",
    diagram: () => diagramTrajectory("ctrl"),
    principle:
      "用 B 样条控制点参数化轨迹——凸包性质使安全性可以由控制点保证；预计算 ESDF（到最近障碍的欧氏距离场），优化中用其梯度把控制点推离障碍，同时最小化控制点差分的平滑项。",
    traits: ["距离场梯度信息丰富，优化收敛快", "适合高频在线重规划", "局部优化，可能陷入局部极小"],
    demo: "完整三次均匀 B 样条控制点优化：平滑/导向/ESDF 障碍/控制点间距约束 + 净空投影。",
  },
  B05: {
    status: "已实现",
    diagram: () => diagramTrajectory("speed"),
    principle:
      "香港科大 Fast-Planner：动力学可行的前端搜索（kinodynamic A*）+ B 样条 ESDF 后端优化 + 时间重分配（急弯段自动降速），构成未知环境在线重规划的完整管线。",
    traits: ["前后端解耦的经典系统架构", "毫秒级在线重规划", "需要实时维护 ESDF"],
    demo: "Kinodynamic A* 前端 + B 样条 ESDF 后端优化，并按曲率与净空做时间重分配。",
  },
  B06: {
    status: "已实现",
    diagram: () => diagramTrajectory("sparse"),
    principle:
      "浙大 MINCO：用少量中间路标 + 每段固定为“给定边界条件下的解析最优多项式”来参数化轨迹，把时空联合优化降维到路标位置与段时长上，梯度可解析回传，效率与质量兼得。",
    traits: ["时间与空间联合优化", "参数稀疏、收敛快，支持多种约束", "GCOPTER 等系统的核心"],
    demo: "MINCO 风格稀疏路标与段时长联合优化 + 闭式 minimum-snap 多项式重建。",
  },
  C01: {
    status: "已实现",
    diagram: () => diagramVelocityObstacle("orca"),
    principle:
      "对每个邻机在速度空间构造速度障碍并取其边界上距当前相对速度最近的点，得到一条“各自负担一半修正量”的半平面约束（ORCA 线）；所有邻机约束的交集内选最接近期望速度的速度，是一个小线性规划。",
    traits: ["去中心化、无需通信，理论上成对无碰", "线性规划高效，支持上千智能体", "拥挤 / 死锁场景需要额外机制"],
    demo: "完整 3D ORCA：为每个邻机生成 ORCA 半平面，并用 Dykstra 投影求最接近期望速度的可行解。",
  },
  C02: {
    status: "已实现",
    diagram: () => diagramVelocityObstacle("rvo"),
    principle:
      "速度障碍 (VO) 假设对方速度不变，双方同时按 VO 避让会来回振荡。RVO 把锥顶从对方速度 vB 移到双方均值 (vA+vB)/2——相当于假设对方也承担一半避让责任，消除了振荡。",
    traits: ["互惠假设消除 VO 的振荡问题", "几何直观：速度选在锥外即安全", "多邻居时约束可能互相冲突"],
    demo: "完整 RVO 3D：构造截断互惠速度障碍，迭代边界投影并选择最接近期望速度的安全速度。",
  },
  C03: {
    status: "已实现",
    diagram: () => diagramVoronoiCell(),
    principle:
      "每机计算自己的缓冲 Voronoi 胞：Voronoi 胞向内收缩安全半径后的区域。每步保证下一位置仍在胞内（对每个邻居即一条半平面约束）。相邻机的胞互不相交，因此位置永不重叠，安全性有简洁的几何证明。",
    traits: ["只需邻居位置，不需要速度或意图", "无碰保证的证明非常简洁", "行为保守，拥挤时通行效率低"],
    demo: "缓冲 Voronoi 胞半平面交集投影：Dykstra 循环求离期望速度最近的可行速度。",
  },
  C04: {
    status: "已实现",
    diagram: () => diagramRollout(),
    principle:
      "分布式模型预测控制：每机用邻机广播 / 预测的未来轨迹作约束，滚动求解自己未来 N 步的最优控制序列，只执行第一步，下一帧滑动重解。冲突在预测域内被显式提前化解。",
    traits: ["显式处理未来冲突与动力学约束", "计算量随预测域与邻居数增长", "性能依赖邻机轨迹预测的质量"],
    demo: "完整滚动 DMPC：warm-start 控制序列 + 坐标下降优化，预测域内联合约束邻机、加速度、ESDF 与高度。",
  },
  C05: {
    status: "已实现",
    diagram: () => diagramCommit(),
    principle:
      "MADER（MIT 2020）面向异步通信的去中心化规划：新轨迹必须与所有已收到的邻机“承诺轨迹”无碰（check）才能承诺；承诺间隙里收到别人的新承诺则复查（recheck），失败就放弃新轨迹保留旧承诺——通信延迟下依然安全。",
    traits: ["显式处理异步与通信延迟", "承诺 / 复查机制保证一致性", "需要广播轨迹"],
    demo: "MADER/RMADER 承诺轨迹：候选轨迹 check、异步 recheck、旧承诺回滚与制动承诺回退。",
  },
  C06: {
    status: "已实现",
    diagram: () => diagramDetour(),
    principle:
      "EGO-Swarm（浙大 2021）：无需 ESDF 的梯度局部规划——把邻机广播轨迹当作时变障碍写进 B 样条优化的惩罚项，冲突段沿切向生成绕行梯度；整个集群只靠广播轨迹即可互避，计算极轻。",
    traits: ["无需距离场，单机毫秒级重规划", "分布式、可扩展到大集群", "局部方法，稠密障碍下需要全局引导"],
    demo: "EGO-Swarm 局部 B 样条控制点优化：广播轨迹惩罚 + 拓扑绕行梯度 + ESDF 障碍梯度。",
  },
  C07: {
    status: "已实现",
    diagram: () => diagramStagger(),
    principle:
      "把冲突消解从空间维移到时间维：预测到多机将同时通过同一空间瓶颈时，按优先级或协商给部分机附加起飞 / 通过延迟，错峰通过，空间路径完全不变。",
    traits: ["不改变空间路径，实现与验证都简单", "对瓶颈型冲突效果显著", "以总时间为代价"],
    demo: "完整 DCP 错峰调度：体素-时间与边交换保留表，按优先级追加延迟并二次消解残余冲突。",
  },
  C08: {
    status: "已实现",
    diagram: () => diagramVelocityObstacle("hrvo"),
    principle:
      "混合 RVO：对每对相遇机，把锥顶放在 VO 与 RVO 锥顶之间的偏置位置，使“从期望侧绕行”便宜、“从错误侧绕行”昂贵，从而引导双方选择同侧绕行，解决 RVO 残留的侧向抖动与僵持。",
    traits: ["绕行方向一致性优于 RVO", "仍然无需通信", "几何构造比 RVO 复杂"],
    demo: "完整 3D HRVO：构造混合 VO/RVO 锥顶与期望侧切向偏置，并在速度空间选择最优安全速度。",
  },
  D01: {
    status: "已实现",
    diagram: () => diagramForces(false),
    principle:
      "人工势场（Khatib 1986）：目标产生吸引势，障碍与邻机产生随距离增强的排斥势，机体沿总势场的负梯度运动。只有局部感知与极小计算量，是最早的反应式避障方法。",
    traits: ["实现最简单、实时性最好", "存在局部极小陷阱（凹形障碍、对称布局）", "参数敏感，斥力过强会抖动"],
    demo: "沿全局路径的吸引 + 个体与障碍的二次衰减斥力。",
  },
  D02: {
    status: "已实现",
    diagram: () => diagramBoids(),
    principle:
      "Reynolds 1987 的三条局部规则：分离（远离过近的邻居）、对齐（匹配邻居平均速度）、聚合（靠近邻居质心）。每架个体只看局部邻域，群体层面的协调运动自发涌现。",
    traits: ["涌现式集群行为的开山之作", "三个权重即可调出丰富群体形态", "无全局目标保证，需叠加导航项"],
    demo: "独立实现的三规则（各自权重可配）+ 全局路径导航项。",
  },
  D03: {
    status: "已实现",
    diagram: () => diagramLattice(),
    principle:
      "Olfati-Saber 2006 给出带收敛性证明的 flocking 控制律：用 σ-norm 平滑距离度量与 bump 函数构造邻接权重，有界作用函数 φ 产生“近推远拉”的梯度项（收敛到等间距 α-lattice），加上速度一致项与导航项。",
    traits: ["有 Lyapunov 收敛性分析的理论方法", "参数物理含义清晰（间距 d、感知半径 r）", "力有界，稠密拥挤下不保证避碰"],
    demo: "完整 α-lattice flocking：σ-norm 梯度项、速度一致项、导航项与 beta-agent 障碍项。",
  },
  D04: {
    status: "已实现",
    diagram: () => diagramBraking(),
    principle:
      "Vásárhelyi 2018（30 架实机户外验证）：短程线性排斥 + 基于理想制动曲线 D(r) 的速度对齐——两机距离越近，允许的速度差越小，超出即产生摩擦力把速度拉齐，显式吸收真实飞行器的惯性、延迟与噪声。",
    traits: ["面向实机干扰设计，鲁棒性强", "制动曲线显式处理惯性约束", "参数较多（原文用进化算法整定）"],
    demo: "完整 Vásárhelyi 控制律：短程排斥、制动曲线摩擦、前向各向异性权重与自推进项。",
  },
  D05: {
    status: "已实现",
    diagram: () => diagramForces(true),
    principle:
      "Helbing 社会力模型移植到 3D：期望速度驱动力 (v* − v)/τ 让个体回到目标速度，个体间与障碍的指数衰减斥力 A·e^((r−d)/B) 表达“心理安全距离”，近距离陡增、远距离温和。",
    traits: ["行人 / 群体流动建模的经典模型", "指数斥力的软硬边界特性良好", "原为二阶（力→加速度）模型"],
    demo: "驱动力 + 个体指数斥力 + 障碍指数斥力三项完整实现（速度级积分近似）。",
  },
  E01: {
    status: "已实现",
    diagram: () => diagramPolicy("blend"),
    principle:
      "GLAS（Rivière 2020）：用全局规划器批量生成示教数据，模仿学习出只依赖局部观测的分布式策略，部署时策略输出与安全备份动作按危险度 λ 混合：u = (1−λ)·u_π + λ·u_safe，兼顾全局知识与安全保证。",
    traits: ["离线学到全局协调知识，在线只需局部观测", "安全模块提供可证明的兜底", "依赖示教数据的覆盖度"],
    demo: "固定权重 GLAS 风格局部观测网络：k 近邻注意力聚合输出策略动作，并按危险度 λ 与安全备份动作混合。",
  },
  E02: {
    status: "已实现",
    diagram: () => diagramPolicy("discrete"),
    principle:
      "PRIMAL（Sartoretti 2019）：强化学习 (A3C) 与模仿学习 (ODrM*) 混合训练分布式 MAPF 策略——每机从局部视野的网格观测中选择离散移动动作，“给别人让路”等协作行为从训练中涌现。",
    traits: ["在线推理极快，规模近似线性扩展", "离散网格动作空间", "无最优与完备性保证"],
    demo: "PRIMAL/PRIMAL2 风格离散策略：局部观测编码 + 固定权重动作评分网络，在 26 个 3D 移动动作与等待动作中选择。",
  },
  E03: {
    status: "已实现",
    diagram: () => diagramPolicy("filter"),
    principle:
      "控制屏障函数 (CBF)：h(x) ≥ 0 定义安全集，只要控制满足 ḣ ≥ −αh 系统就永不出界。Neural CBF 用网络学习 h，部署时对任意上游策略解一个小 QP：在 CBF 约束下最小改动策略输出——安全滤波器。",
    traits: ["对任意上游策略提供安全保证", "QP 逐约束可解析投影，代价小", "难点在 h 的构造 / 学习"],
    demo: "Neural CBF 风格滤波：固定权重网络估计屏障裕度与 α，再对邻机/障碍约束做两轮 QP 顺序投影。",
  },
  E04: {
    status: "已实现",
    diagram: () => diagramPolicy("safety"),
    principle:
      "把安全责任从策略中剥离：RL 策略专注任务性能，输出动作先经过独立安全层——预测未来短时域内是否碰撞，若是则修正或抑制该动作再执行。训练与安全解耦，各自可独立迭代。",
    traits: ["责任分离，训练可专注性能", "安全层可能牺牲策略最优性", "安全层本身的覆盖度是关键"],
    demo: "固定权重 RL 策略网络先输出动作，安全层对候选动作做短时 rollout 与价值评估，选择满足安全约束的动作。",
  },
  E05: {
    status: "已实现",
    diagram: () => diagramPolicy("mlp"),
    principle:
      "端到端集群强化学习：邻机相对状态等观测直接映射到控制输出，中间不设人工规则，协作与避碰行为完全由奖励函数塑造。集中训练、分布执行 (CTDE) 是常见范式。",
    traits: ["表达能力最强，无手工设计偏置", "训练成本高，泛化与安全难保证", "部署常需蒸馏 / 剪裁"],
    demo: "端到端 CTDE MLP 推理：k 近邻相对状态、目标方向与速度观测直接映射为油门和三轴速度修正，并叠加分离保护。",
  },
  F01: {
    status: "已实现",
    diagram: () => diagramGantt(),
    principle:
      "多智能体路径规划 (MAPF) 的一般形式：在空间 × 时间图上为全体智能体求互不冲突（不同时占同一顶点 / 不对穿同一条边）的路径集合。经典解法族包括联合 A*、CBS、优先级规划与保留表法。",
    traits: ["集中式全局视角，可做到最优", "联合状态空间随机数指数增长", "仓储、编队等结构化场景的标准问题"],
    demo: "完整 3D MAPF 保留表：体素-时间顶点约束 + 边交换约束，按时序规划无冲突通行。",
  },
  F02: {
    status: "已实现",
    diagram: () => diagramGantt("conflict"),
    principle:
      "冲突驱动搜索 (CBS)：底层为每机独立 A*，顶层维护约束树——发现两机冲突就分支成两个子问题（各禁止一方在该时刻占该格）分别重规划，直到无冲突，可证最优。ECBS 允许有界次优以换取速度。",
    traits: ["最优且通常远快于联合搜索", "冲突密集时约束树可能爆炸", "ECBS / 加权变体是工程主力"],
    demo: "CBS/ECBS 约束树：检测最早顶点/边冲突并二分延迟约束，使用 focal bound 控制搜索量。",
  },
  F03: {
    status: "已实现",
    diagram: () => diagramGantt("priority"),
    principle:
      "基于优先级的搜索 (PBS)：给智能体排优先序，低优先级把所有高优先级的轨迹当作动态障碍依次规划。PBS 在优先序导致失败时对“谁让谁”做二分分支，按需探索优先序空间。",
    traits: ["比 CBS 快得多，适合大规模编队", "牺牲最优性", "固定优先序可能死锁，需要回溯"],
    demo: "PBS 优先级搜索：冲突触发优先级约束，拓扑排序后用保留表顺序规划。",
  },
  F04: {
    status: "已实现",
    diagram: () => diagramJoint(),
    principle:
      "序列凸规划 (SCP)：机间避碰约束非凸，无法直接用凸优化；SCP 在当前解附近把它线性化（凸化），解凸 QP 更新全体轨迹，再重新线性化迭代，直至收敛。可集中生成高密度编队变换轨迹。",
    traits: ["全体轨迹联合优化，队形变换丝滑", "集中式，规模与实时性受限", "收敛到局部最优"],
    demo: "完整 SCP：时间采样轨迹、线性化机间分离约束、平滑/路径跟踪目标与 ESDF 障碍投影迭代求解。",
  },
  F05: {
    status: "已实现",
    diagram: () => diagramAssign(),
    principle:
      "任务分配层：“谁去哪个目标”建模为二分图最小代价完美匹配。匈牙利算法 O(n³) 给出精确解；拍卖算法让每机对净收益最高的目标竞价、价格上涨直到无人愿意换，天然支持分布式与增量式执行。",
    traits: ["最优分配可显著缩短总航程", "拍卖算法可分布式、可在线增量", "需与底层路径规划器配合"],
    demo: "精确 Hungarian 分配（中小规模）+ ε-拍卖后备（大规模）+ 到分配目标的单目标 A* 路由。",
  },
};

function renderAlgorithmInfo() {
  const algorithm = getAlgorithm();
  const details = algorithmDetails[algorithm.id];
  const panel = document.querySelector("#infoPanel");
  if (!details || !panel) return;
  document.querySelector("#infoCode").textContent = algorithm.id;
  document.querySelector("#infoName").textContent = algorithm.name;
  document.querySelector("#infoFamily").textContent = algorithm.familyTitle;
  const status = document.querySelector("#infoStatus");
  status.textContent = details.status;
  status.className = `info-status${details.status === "已实现" ? " full" : details.status === "部分实现" ? " partial" : ""}`;
  document.querySelector("#infoDiagram").innerHTML = details.diagram();
  document.querySelector("#infoPrinciple").textContent = details.principle;
  document.querySelector("#infoTraits").innerHTML = details.traits.map((trait) => `<li>${trait}</li>`).join("");
  document.querySelector("#infoDemo").textContent = details.demo;
}

const modeTint = {
  search: new THREE.Color("#2ee6d6"),
  optimize: new THREE.Color("#76e06f"),
  avoid: new THREE.Color("#ffbf47"),
  field: new THREE.Color("#4fc3ff"),
  learning: new THREE.Color("#b692ff"),
  central: new THREE.Color("#ff7a90"),
};

const agentBaseColor = new THREE.Color("#6fe7ff");
const agentHighlightColor = new THREE.Color("#f2fdff");
const agentEmissiveColor = new THREE.Color("#3bdcff");
const agentGlowColor = new THREE.Color("#42dfff");

function getAlgorithm() {
  return algorithmById.get(state.algorithmId) ?? algorithms[0];
}

function getFamily(mode = state.mode) {
  return algorithmGroups.find((group) => group.mode === mode) ?? algorithmGroups[0];
}

function getProfile() {
  return getAlgorithm().profile ?? {};
}

function firstAlgorithmIdForMode(mode) {
  return getFamily(mode).algorithms[0].id;
}

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: "high-performance",
});
renderer.setClearColor(0x0b0c0f, 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0b0c0f, 0.018);

const camera = new THREE.PerspectiveCamera(54, 1, 0.1, 900);
camera.position.set(38, 34, 45);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 6, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 18;
controls.maxDistance = 115;

const world = new THREE.Group();
const obstacleGroup = new THREE.Group();
const pathGroup = new THREE.Group();
const overlayGroup = new THREE.Group();
const corridorGroup = new THREE.Group();
const selectedGroup = new THREE.Group();
scene.add(world, obstacleGroup, pathGroup, overlayGroup, corridorGroup, selectedGroup);

const ambient = new THREE.HemisphereLight(0xd7fff8, 0x16100d, 1.8);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(18, 30, 16);
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0xffc47a, 0.85);
rimLight.position.set(-28, 18, -20);
scene.add(rimLight);

const floorMaterial = new THREE.MeshStandardMaterial({
  color: 0x171a1e,
  metalness: 0.12,
  roughness: 0.82,
});
const floor = new THREE.Mesh(new THREE.PlaneGeometry(72, 72, 1, 1), floorMaterial);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
world.add(floor);

const padMaterialA = new THREE.MeshBasicMaterial({
  color: 0x2ee6d6,
  transparent: true,
  opacity: 0.2,
  side: THREE.DoubleSide,
});
const padMaterialB = new THREE.MeshBasicMaterial({
  color: 0xff667c,
  transparent: true,
  opacity: 0.22,
  side: THREE.DoubleSide,
});
const startPad = new THREE.Mesh(new THREE.CircleGeometry(5.5, 48), padMaterialA);
startPad.rotation.x = -Math.PI / 2;
startPad.position.set(-30, 0.03, 0);
const goalPad = new THREE.Mesh(new THREE.CircleGeometry(5.5, 48), padMaterialB);
goalPad.rotation.x = -Math.PI / 2;
goalPad.position.set(30, 0.04, 0);
world.add(startPad, goalPad);

const droneGeometry = new THREE.ConeGeometry(0.42, 1.22, 12, 1);
droneGeometry.rotateX(Math.PI / 2);
const droneMaterial = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  metalness: 0.08,
  roughness: 0.22,
  emissive: agentEmissiveColor,
  emissiveIntensity: 0.62,
  vertexColors: true,
});
let droneMesh = new THREE.InstancedMesh(droneGeometry, droneMaterial, state.count);
droneMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
droneMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(state.count * 3), 3);
scene.add(droneMesh);

const glowGeometry = new THREE.SphereGeometry(0.74, 16, 8);
const glowMaterial = new THREE.MeshBasicMaterial({
  color: agentGlowColor,
  transparent: true,
  opacity: 0.26,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  vertexColors: true,
});
let glowMesh = new THREE.InstancedMesh(glowGeometry, glowMaterial, state.count);
glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
glowMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(state.count * 3), 3);
glowMesh.frustumCulled = false;
scene.add(glowMesh);

const safetyGeometry = new THREE.SphereGeometry(1, 14, 8);
const safetyMaterial = new THREE.MeshBasicMaterial({
  color: 0xffbf47,
  transparent: true,
  opacity: 0.075,
  wireframe: true,
});
let safetyMesh = new THREE.InstancedMesh(safetyGeometry, safetyMaterial, state.count);
safetyMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(safetyMesh);

const selectedMaterial = new THREE.MeshBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.9,
  wireframe: true,
});
const selectedRing = new THREE.Mesh(new THREE.TorusGeometry(0.72, 0.025, 8, 40), selectedMaterial);
selectedRing.rotation.x = Math.PI / 2;
selectedGroup.add(selectedRing);

let pathLines = null;
let trailLines = null;
let vectorLines = null;
let gridLines = null;
let searchCloud = null;
let selectedPathLine = null;
let selectedRawLine = null;

const tempMatrix = new THREE.Matrix4();
const tempQuaternion = new THREE.Quaternion();
const identityQuaternion = new THREE.Quaternion();
const tempScale = new THREE.Vector3(1, 1, 1);
const forward = new THREE.Vector3(0, 0, 1);
const tempVector = new THREE.Vector3();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const clock = new THREE.Clock();

const ui = {
  fps: document.querySelector("#fpsValue"),
  conflict: document.querySelector("#conflictValue"),
  distance: document.querySelector("#distanceValue"),
  selectedId: document.querySelector("#selectedId"),
  selectedMode: document.querySelector("#selectedMode"),
  selectedSpeed: document.querySelector("#selectedSpeed"),
  pauseButton: document.querySelector("#pauseButton"),
  pauseIcon: document.querySelector("#pauseIcon"),
  resetButton: document.querySelector("#resetButton"),
  sceneSelect: document.querySelector("#sceneSelect"),
  algorithmButtons: document.querySelector("#algorithmButtons"),
  algorithmFamilyLabel: document.querySelector("#algorithmFamilyLabel"),
};

function createRng(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function scenarioSeed(name) {
  let seed = 2166136261;
  for (let i = 0; i < name.length; i += 1) {
    seed ^= name.charCodeAt(i);
    seed = Math.imul(seed, 16777619);
  }
  return seed >>> 0;
}

class MinHeap {
  constructor() {
    this.items = [];
  }

  push(node, priority) {
    const item = { node, priority };
    this.items.push(item);
    this.bubbleUp(this.items.length - 1);
  }

  pop() {
    if (this.items.length === 0) return null;
    const top = this.items[0];
    const end = this.items.pop();
    if (this.items.length > 0) {
      this.items[0] = end;
      this.sinkDown(0);
    }
    return top;
  }

  bubbleUp(index) {
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.items[parent].priority <= this.items[index].priority) break;
      [this.items[parent], this.items[index]] = [this.items[index], this.items[parent]];
      index = parent;
    }
  }

  sinkDown(index) {
    const length = this.items.length;
    while (true) {
      const left = index * 2 + 1;
      const right = left + 1;
      let smallest = index;
      if (left < length && this.items[left].priority < this.items[smallest].priority) {
        smallest = left;
      }
      if (right < length && this.items[right].priority < this.items[smallest].priority) {
        smallest = right;
      }
      if (smallest === index) break;
      [this.items[index], this.items[smallest]] = [this.items[smallest], this.items[index]];
      index = smallest;
    }
  }
}

function makeGrid() {
  const grid = {
    nx: 34,
    ny: 8,
    nz: 34,
    cell: 2,
    yStep: 1.65,
    yBase: 1.65,
    passable: null,
  };
  grid.total = grid.nx * grid.ny * grid.nz;
  grid.index = (x, y, z) => (y * grid.nz + z) * grid.nx + x;
  grid.unpack = (index) => {
    const x = index % grid.nx;
    const z = Math.floor(index / grid.nx) % grid.nz;
    const y = Math.floor(index / (grid.nx * grid.nz));
    return { x, y, z };
  };
  grid.cellToWorld = (x, y, z) =>
    new THREE.Vector3(
      (x - grid.nx / 2 + 0.5) * grid.cell,
      grid.yBase + y * grid.yStep,
      (z - grid.nz / 2 + 0.5) * grid.cell,
    );
  grid.worldToCell = (point) => ({
    x: clamp(Math.floor(point.x / grid.cell + grid.nx / 2), 0, grid.nx - 1),
    y: clamp(Math.round((point.y - grid.yBase) / grid.yStep), 0, grid.ny - 1),
    z: clamp(Math.floor(point.z / grid.cell + grid.nz / 2), 0, grid.nz - 1),
  });
  return grid;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material.dispose());
    } else {
      child.material?.dispose();
    }
  }
}

function addObstacle(kind, options) {
  const color = options.color ?? 0x59636d;
  let geometry;
  let mesh;
  const material = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.08,
    roughness: 0.86,
    transparent: true,
    opacity: options.opacity ?? 0.78,
  });

  if (kind === "box") {
    geometry = new THREE.BoxGeometry(options.size.x, options.size.y, options.size.z);
    mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(options.center);
  } else if (kind === "cylinder") {
    geometry = new THREE.CylinderGeometry(options.radius, options.radius, options.height, 18);
    mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(options.center);
  } else {
    geometry = new THREE.SphereGeometry(options.radius, 24, 14);
    mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(options.center);
  }

  obstacleGroup.add(mesh);
  const obstacle = { kind, mesh, ...options };
  state.obstacles.push(obstacle);
  return obstacle;
}

function buildScenario(name) {
  clearGroup(obstacleGroup);
  state.obstacles = [];

  const cyanGlass = 0x355862;
  const amberStone = 0x7b6742;
  const roseGate = 0x7d3b4a;
  const greenPillar = 0x426d58;
  const violetVolume = 0x574a78;

  if (name === "urban") {
    const blocks = [
      [-18, -14, 4, 5, 10],
      [-10, -8, 5, 4, 7],
      [-4, -16, 4, 6, 9],
      [4, -10, 5, 5, 12],
      [12, -15, 5, 6, 8],
      [-16, 10, 6, 4, 11],
      [-7, 15, 4, 5, 8],
      [3, 9, 5, 4, 9],
      [12, 13, 5, 5, 12],
      [19, 3, 4, 7, 8],
      [-1, 0, 5, 5, 6],
    ];
    blocks.forEach(([x, z, w, d, h], index) => {
      addObstacle("box", {
        center: new THREE.Vector3(x, h / 2, z),
        size: new THREE.Vector3(w, h, d),
        color: index % 2 ? cyanGlass : amberStone,
      });
    });
    addObstacle("sphere", {
      center: new THREE.Vector3(7, 9, -1),
      radius: 3.1,
      color: roseGate,
      opacity: 0.48,
    });
  }

  if (name === "canyon") {
    for (let i = -16; i <= 16; i += 8) {
      addObstacle("box", {
        center: new THREE.Vector3(i, 6.2, -10),
        size: new THREE.Vector3(6.5, 12.4, 8.5),
        color: amberStone,
      });
      addObstacle("box", {
        center: new THREE.Vector3(i, 6.2, 10),
        size: new THREE.Vector3(6.5, 12.4, 8.5),
        color: amberStone,
      });
    }
    addObstacle("box", {
      center: new THREE.Vector3(0, 3.4, 0),
      size: new THREE.Vector3(5.5, 6.8, 4.8),
      color: roseGate,
      opacity: 0.62,
    });
    addObstacle("box", {
      center: new THREE.Vector3(13, 8.2, 0),
      size: new THREE.Vector3(4.5, 7, 6),
      color: cyanGlass,
      opacity: 0.66,
    });
  }

  if (name === "forest") {
    const rng = createRng(42);
    for (let i = 0; i < 38; i += 1) {
      const x = -21 + rng() * 43;
      const z = -22 + rng() * 44;
      if (Math.abs(x) < 4 && Math.abs(z) < 5) continue;
      const height = 7 + rng() * 6;
      const radius = 0.65 + rng() * 0.65;
      addObstacle("cylinder", {
        center: new THREE.Vector3(x, height / 2, z),
        radius,
        height,
        color: greenPillar,
        opacity: 0.78,
      });
    }
    addObstacle("sphere", {
      center: new THREE.Vector3(1, 8, 0),
      radius: 2.8,
      color: violetVolume,
      opacity: 0.42,
    });
  }

  if (name === "multilevel") {
    const volumes = [
      [-16, 3, -12, 8, 3, 7],
      [-8, 8, 4, 6, 3.2, 8],
      [0, 4, -3, 7, 2.8, 5],
      [8, 10, 10, 7, 3, 7],
      [17, 5, -8, 6, 3.2, 8],
      [2, 13, -16, 10, 2.8, 5],
    ];
    volumes.forEach(([x, y, z, w, h, d], index) => {
      addObstacle("box", {
        center: new THREE.Vector3(x, y, z),
        size: new THREE.Vector3(w, h, d),
        color: index % 2 ? violetVolume : cyanGlass,
        opacity: 0.54,
      });
    });
    addObstacle("box", {
      center: new THREE.Vector3(-1, 2, 10),
      size: new THREE.Vector3(22, 4, 3.5),
      color: amberStone,
      opacity: 0.72,
    });
  }
}

function isWorldBlocked(point, margin = 0.58) {
  if (point.y < 0.75 || point.y > 15.2) return true;
  for (const obstacle of state.obstacles) {
    if (obstacle.kind === "box") {
      const s = obstacle.size;
      const c = obstacle.center;
      if (
        Math.abs(point.x - c.x) <= s.x / 2 + margin &&
        Math.abs(point.y - c.y) <= s.y / 2 + margin &&
        Math.abs(point.z - c.z) <= s.z / 2 + margin
      ) {
        return true;
      }
    } else if (obstacle.kind === "cylinder") {
      const dx = point.x - obstacle.center.x;
      const dz = point.z - obstacle.center.z;
      const horizontal = Math.sqrt(dx * dx + dz * dz);
      if (
        horizontal <= obstacle.radius + margin &&
        Math.abs(point.y - obstacle.center.y) <= obstacle.height / 2 + margin
      ) {
        return true;
      }
    } else {
      if (point.distanceTo(obstacle.center) <= obstacle.radius + margin) {
        return true;
      }
    }
  }
  return false;
}

function rebuildGrid() {
  const grid = makeGrid();
  grid.passable = new Uint8Array(grid.total);
  for (let y = 0; y < grid.ny; y += 1) {
    for (let z = 0; z < grid.nz; z += 1) {
      for (let x = 0; x < grid.nx; x += 1) {
        const point = grid.cellToWorld(x, y, z);
        grid.passable[grid.index(x, y, z)] = isWorldBlocked(point, 0.72) ? 0 : 1;
      }
    }
  }
  state.grid = grid;
  state.goals = buildGoals(grid);
  state.distance = computeDistanceField(grid, state.goals);
  goalSet = new Set(state.goals.map((goal) => goal.index));
  computeEsdf();
  rebuildGridOverlay();
  rebuildSearchCloud();
}

function buildGoals(grid) {
  const goals = [];
  for (let y = 1; y < grid.ny - 1; y += 1) {
    for (let z = 5; z < grid.nz - 5; z += 3) {
      const x = grid.nx - 3;
      const index = grid.index(x, y, z);
      if (grid.passable[index]) goals.push({ x, y, z, index });
    }
  }
  if (goals.length) return goals;
  for (let y = 0; y < grid.ny; y += 1) {
    for (let z = 0; z < grid.nz; z += 1) {
      const x = grid.nx - 2;
      const index = grid.index(x, y, z);
      if (grid.passable[index]) goals.push({ x, y, z, index });
    }
  }
  return goals;
}

const neighborOffsets = [];
for (let dy = -1; dy <= 1; dy += 1) {
  for (let dz = -1; dz <= 1; dz += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (dx === 0 && dy === 0 && dz === 0) continue;
      neighborOffsets.push({
        dx,
        dy,
        dz,
        cost: Math.hypot(dx, dy * 1.25, dz) + Math.abs(dy) * 0.14,
      });
    }
  }
}

function computeDistanceField(grid, goals) {
  const distance = new Float32Array(grid.total);
  distance.fill(Number.POSITIVE_INFINITY);
  const heap = new MinHeap();
  for (const goal of goals) {
    distance[goal.index] = 0;
    heap.push(goal.index, 0);
  }

  while (heap.items.length) {
    const current = heap.pop();
    if (!current || current.priority !== distance[current.node]) continue;
    const c = grid.unpack(current.node);
    for (const offset of neighborOffsets) {
      const nx = c.x + offset.dx;
      const ny = c.y + offset.dy;
      const nz = c.z + offset.dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) continue;
      const next = grid.index(nx, ny, nz);
      if (!grid.passable[next]) continue;
      const nextDistance = current.priority + offset.cost;
      if (nextDistance < distance[next]) {
        distance[next] = nextDistance;
        heap.push(next, nextDistance);
      }
    }
  }

  return distance;
}

function nearestFreeCell(cell, requireFinite = true) {
  const grid = state.grid;
  const distance = state.distance;
  let best = null;
  let bestScore = Number.POSITIVE_INFINITY;
  for (let radius = 0; radius <= 8; radius += 1) {
    for (let y = cell.y - radius; y <= cell.y + radius; y += 1) {
      for (let z = cell.z - radius; z <= cell.z + radius; z += 1) {
        for (let x = cell.x - radius; x <= cell.x + radius; x += 1) {
          if (x < 0 || y < 0 || z < 0 || x >= grid.nx || y >= grid.ny || z >= grid.nz) continue;
          const index = grid.index(x, y, z);
          if (!grid.passable[index]) continue;
          if (requireFinite && !Number.isFinite(distance[index])) continue;
          const score = Math.hypot(x - cell.x, (y - cell.y) * 1.4, z - cell.z);
          if (score < bestScore) {
            bestScore = score;
            best = { x, y, z, index };
          }
        }
      }
    }
    if (best) return best;
  }
  return { x: 1, y: 2, z: Math.floor(grid.nz / 2), index: grid.index(1, 2, Math.floor(grid.nz / 2)) };
}

function tracePathFromCell(startCell) {
  const grid = state.grid;
  const distance = state.distance;
  let current = nearestFreeCell(startCell, true);
  const cells = [current];
  const used = new Set([current.index]);

  for (let steps = 0; steps < 170; steps += 1) {
    const currentDistance = distance[current.index];
    if (!Number.isFinite(currentDistance) || currentDistance <= 0.001) break;
    let best = null;
    let bestValue = currentDistance;
    for (const offset of neighborOffsets) {
      const nx = current.x + offset.dx;
      const ny = current.y + offset.dy;
      const nz = current.z + offset.dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) continue;
      const index = grid.index(nx, ny, nz);
      if (!grid.passable[index] || used.has(index)) continue;
      const candidate = distance[index] + offset.cost * 0.02;
      if (candidate < bestValue) {
        bestValue = candidate;
        best = { x: nx, y: ny, z: nz, index };
      }
    }
    if (!best) break;
    cells.push(best);
    used.add(best.index);
    current = best;
  }

  const raw = cells.map((cell) => grid.cellToWorld(cell.x, cell.y, cell.z));
  const simplified = simplifyCells(cells).map((cell) => grid.cellToWorld(cell.x, cell.y, cell.z));
  const smooth = smoothPath(simplified);
  return {
    cells,
    raw,
    smooth: smooth.length > 1 ? smooth : raw,
  };
}

function simplifyCells(cells) {
  if (cells.length <= 2) return cells.slice();
  const simplified = [cells[0]];
  let i = 0;
  while (i < cells.length - 1) {
    let best = i + 1;
    const end = Math.min(cells.length - 1, i + 16);
    for (let j = end; j > i + 1; j -= 1) {
      if (lineOfSight(cells[i], cells[j])) {
        best = j;
        break;
      }
    }
    simplified.push(cells[best]);
    i = best;
  }
  return simplified;
}

function lineOfSight(a, b) {
  const grid = state.grid;
  const steps = Math.ceil(Math.hypot(a.x - b.x, (a.y - b.y) * 1.2, a.z - b.z) * 2);
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 0 : i / steps;
    const x = Math.round(THREE.MathUtils.lerp(a.x, b.x, t));
    const y = Math.round(THREE.MathUtils.lerp(a.y, b.y, t));
    const z = Math.round(THREE.MathUtils.lerp(a.z, b.z, t));
    if (x < 0 || y < 0 || z < 0 || x >= grid.nx || y >= grid.ny || z >= grid.nz) return false;
    if (!grid.passable[grid.index(x, y, z)]) return false;
  }
  return true;
}

function smoothPath(points) {
  if (points.length <= 2) return points.map((point) => point.clone());
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.35);
  let length = 0;
  for (let i = 0; i < points.length - 1; i += 1) length += points[i].distanceTo(points[i + 1]);
  const samples = Math.min(220, Math.max(points.length * 5, Math.ceil(length / 1.1), 16));
  return curve.getPoints(samples).map((point) => point.clone());
}

// ===== Per-algorithm global planners and trajectory optimizers =====

let goalSet = new Set();
let esdf = null;

const offsetUnits = neighborOffsets.map((offset) =>
  new THREE.Vector3(offset.dx, offset.dy, offset.dz).normalize(),
);

function computeEsdf() {
  const grid = state.grid;
  const field = new Float32Array(grid.total);
  field.fill(Number.POSITIVE_INFINITY);
  const heap = new MinHeap();
  for (let i = 0; i < grid.total; i += 1) {
    if (!grid.passable[i]) {
      field[i] = 0;
      heap.push(i, 0);
    }
  }
  while (heap.items.length) {
    const current = heap.pop();
    if (!current || current.priority !== field[current.node]) continue;
    const c = grid.unpack(current.node);
    for (const offset of neighborOffsets) {
      const nx = c.x + offset.dx;
      const ny = c.y + offset.dy;
      const nz = c.z + offset.dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) continue;
      const next = grid.index(nx, ny, nz);
      const stepWorld = Math.hypot(offset.dx * grid.cell, offset.dy * grid.yStep, offset.dz * grid.cell);
      const nextDistance = current.priority + stepWorld;
      if (nextDistance < field[next]) {
        field[next] = nextDistance;
        heap.push(next, nextDistance);
      }
    }
  }
  esdf = field;
}

function sampleEsdf(point) {
  if (!esdf) return 10;
  const grid = state.grid;
  const cell = grid.worldToCell(point);
  const value = esdf[grid.index(cell.x, cell.y, cell.z)];
  return Number.isFinite(value) ? value : 10;
}

function esdfGradient(point, out) {
  const grid = state.grid;
  const cell = grid.worldToCell(point);
  const at = (x, y, z) => {
    const value = esdf[grid.index(clamp(x, 0, grid.nx - 1), clamp(y, 0, grid.ny - 1), clamp(z, 0, grid.nz - 1))];
    return Number.isFinite(value) ? value : 10;
  };
  out.set(
    at(cell.x + 1, cell.y, cell.z) - at(cell.x - 1, cell.y, cell.z),
    at(cell.x, cell.y + 1, cell.z) - at(cell.x, cell.y - 1, cell.z),
    at(cell.x, cell.y, cell.z + 1) - at(cell.x, cell.y, cell.z - 1),
  );
  if (out.lengthSq() < 0.0001) out.set(0, 1, 0);
  return out.normalize();
}

function distanceFieldAt(point) {
  const grid = state.grid;
  const cell = grid.worldToCell(point);
  const value = state.distance[grid.index(cell.x, cell.y, cell.z)];
  return Number.isFinite(value) ? value : 500;
}

const latticeSearch = { gen: 0, stamp: null, gScore: null, parent: null, size: 0 };

function ensureSearchArrays(size) {
  if (latticeSearch.size < size) {
    latticeSearch.stamp = new Int32Array(size);
    latticeSearch.gScore = new Float32Array(size);
    latticeSearch.parent = new Int32Array(size);
    latticeSearch.size = size;
  }
  latticeSearch.gen += 1;
  return latticeSearch;
}

function cellsFromStateChain(found, dims) {
  const grid = state.grid;
  const cells = [];
  let walk = found;
  let guard = 800;
  while (walk >= 0 && guard-- > 0) {
    const cellIndex = Math.floor(walk / dims);
    const c = grid.unpack(cellIndex);
    if (!cells.length || cells[cells.length - 1].index !== cellIndex) {
      cells.push({ x: c.x, y: c.y, z: c.z, index: cellIndex });
    }
    walk = latticeSearch.parent[walk];
  }
  cells.reverse();
  return cells;
}

function planGridAStar(start, goals = goalSet, goalCell = null) {
  const grid = state.grid;
  if (!goals.size) return null;
  const goalX = grid.nx - 3;
  const heuristic = goalCell
    ? (x, y, z) => Math.hypot(goalCell.x - x, (goalCell.y - y) * 1.25, goalCell.z - z)
    : (x) => Math.max(0, goalX - x);
  const s = ensureSearchArrays(grid.total);
  const heap = new MinHeap();
  s.stamp[start.index] = s.gen;
  s.gScore[start.index] = 0;
  s.parent[start.index] = -1;
  heap.push(start.index, heuristic(start.x, start.y, start.z));
  let found = -1;
  let expansions = 0;
  while (heap.items.length) {
    const current = heap.pop();
    if (!current) break;
    const index = current.node;
    const c = grid.unpack(index);
    const g = s.gScore[index];
    if (current.priority > g + heuristic(c.x, c.y, c.z) + 0.001) continue;
    if (goals.has(index)) {
      found = index;
      break;
    }
    if ((expansions += 1) > 40000) break;
    for (const offset of neighborOffsets) {
      const nx = c.x + offset.dx;
      const ny = c.y + offset.dy;
      const nz = c.z + offset.dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) continue;
      const next = grid.index(nx, ny, nz);
      if (!grid.passable[next]) continue;
      const ng = g + offset.cost;
      if (s.stamp[next] !== s.gen || ng < s.gScore[next] - 1e-6) {
        s.stamp[next] = s.gen;
        s.gScore[next] = ng;
        s.parent[next] = index;
        heap.push(next, ng + heuristic(nx, ny, nz));
      }
    }
  }
  if (found < 0) return null;
  return cellsFromStateChain(found, 1);
}

const REST_DIR = neighborOffsets.length;
const directionLookup = new Map(neighborOffsets.map((offset, index) => [`${offset.dx},${offset.dy},${offset.dz}`, index]));
const accelerationOffsets = [{ dx: 0, dy: 0, dz: 0, cost: 0 }, ...neighborOffsets];
const hybridHeadingVectors = Array.from({ length: 16 }, (_, i) => {
  const angle = (i / 16) * Math.PI * 2;
  return { x: Math.cos(angle), z: Math.sin(angle) };
});

function directionIndexFor(dx, dy, dz) {
  if (dx === 0 && dy === 0 && dz === 0) return REST_DIR;
  return directionLookup.get(`${dx},${dy},${dz}`) ?? REST_DIR;
}

function directionOffset(index) {
  return index === REST_DIR ? { dx: 0, dy: 0, dz: 0, cost: 0 } : neighborOffsets[index];
}

function headingIndexToward(fromCell, goalPoint, grid) {
  const from = grid.cellToWorld(fromCell.x, fromCell.y, fromCell.z);
  const angle = Math.atan2(goalPoint.z - from.z, goalPoint.x - from.x);
  return (Math.round(((angle + Math.PI * 2) % (Math.PI * 2)) / (Math.PI * 2 / hybridHeadingVectors.length)) + hybridHeadingVectors.length) % hybridHeadingVectors.length;
}

function headingTurnDistance(a, b, count) {
  const d = Math.abs(a - b);
  return Math.min(d, count - d);
}

function distanceHeuristic(cellIndex) {
  const value = state.distance?.[cellIndex];
  return Number.isFinite(value) ? value : 500;
}

function hybridArcBlocked(grid, cell, heading, nextHeading, nextCell, margin = 0.56) {
  const a = grid.cellToWorld(cell.x, cell.y, cell.z);
  const b = grid.cellToWorld(nextCell.x, nextCell.y, nextCell.z);
  const hv0 = hybridHeadingVectors[heading];
  const hv1 = hybridHeadingVectors[nextHeading];
  const mid = a.clone().lerp(b, 0.5);
  mid.x += (hv0.x - hv1.x) * grid.cell * 0.42;
  mid.z += (hv0.z - hv1.z) * grid.cell * 0.42;
  mid.y = clamp(mid.y, 1.2, 15);
  const point = new THREE.Vector3();
  const steps = Math.max(5, Math.ceil(a.distanceTo(b) / 0.55));
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const omt = 1 - t;
    point.copy(a).multiplyScalar(omt * omt).addScaledVector(mid, 2 * omt * t).addScaledVector(b, t * t);
    if (isWorldBlocked(point, margin)) return true;
  }
  return false;
}

function planHybridAStar(start) {
  const grid = state.grid;
  const goalPoint = goalPointFor(start);
  if (!goalSet.size || !goalPoint) return null;
  const H = hybridHeadingVectors.length;
  const startHeading = headingIndexToward(start, goalPoint, grid);
  const s = ensureSearchArrays(grid.total * H);
  const heap = new MinHeap();
  const startState = start.index * H + startHeading;
  s.stamp[startState] = s.gen;
  s.gScore[startState] = 0;
  s.parent[startState] = -1;
  heap.push(startState, distanceHeuristic(start.index));
  let found = -1;
  let expansions = 0;

  while (heap.items.length) {
    const current = heap.pop();
    if (!current) break;
    const stateIndex = current.node;
    const cellIndex = Math.floor(stateIndex / H);
    const heading = stateIndex % H;
    const c = grid.unpack(cellIndex);
    const g = s.gScore[stateIndex];
    if (current.priority > g + distanceHeuristic(cellIndex) + 0.001) continue;
    if (goalSet.has(cellIndex)) {
      found = stateIndex;
      break;
    }
    if ((expansions += 1) > 95000) break;

    for (let turn = -2; turn <= 2; turn += 1) {
      const nextHeading = (heading + turn + H) % H;
      const hv0 = hybridHeadingVectors[heading];
      const hv1 = hybridHeadingVectors[nextHeading];
      const dx = Math.round((hv0.x + hv1.x) * 0.92);
      const dz = Math.round((hv0.z + hv1.z) * 0.92);
      if (dx === 0 && dz === 0) continue;
      for (const climb of [-1, 0, 1]) {
        const nx = c.x + dx;
        const ny = c.y + climb;
        const nz = c.z + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) continue;
        const nextCellIndex = grid.index(nx, ny, nz);
        if (!grid.passable[nextCellIndex]) continue;
        const nextCell = { x: nx, y: ny, z: nz, index: nextCellIndex };
        if (hybridArcBlocked(grid, c, heading, nextHeading, nextCell)) continue;
        const edgeLength = grid.cellToWorld(c.x, c.y, c.z).distanceTo(grid.cellToWorld(nx, ny, nz)) / grid.cell;
        const turnPenalty = Math.abs(turn) * 0.42 + Math.max(0, Math.abs(turn) - 1) * 0.34;
        const climbPenalty = Math.abs(climb) * 0.38;
        const nextState = nextCellIndex * H + nextHeading;
        const ng = g + edgeLength + turnPenalty + climbPenalty;
        const h = distanceHeuristic(nextCellIndex) + headingTurnDistance(nextHeading, headingIndexToward(nextCell, goalPoint, grid), H) * 0.16;
        if (s.stamp[nextState] !== s.gen || ng < s.gScore[nextState] - 1e-6) {
          s.stamp[nextState] = s.gen;
          s.gScore[nextState] = ng;
          s.parent[nextState] = stateIndex;
          heap.push(nextState, ng + h);
        }
      }
    }
  }
  if (found < 0) return null;
  return cellsFromStateChain(found, H);
}

function stepAllowed(grid, x, y, z, dx, dy, dz) {
  const nx = x + dx;
  const ny = y + dy;
  const nz = z + dz;
  if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) return false;
  if (!grid.passable[grid.index(nx, ny, nz)]) return false;
  const comps = (dx !== 0 ? 1 : 0) + (dy !== 0 ? 1 : 0) + (dz !== 0 ? 1 : 0);
  if (comps >= 2) {
    if (dx !== 0 && !grid.passable[grid.index(x + dx, y, z)]) return false;
    if (dy !== 0 && !grid.passable[grid.index(x, y + dy, z)]) return false;
    if (dz !== 0 && !grid.passable[grid.index(x, y, z + dz)]) return false;
    if (comps === 3) {
      if (!grid.passable[grid.index(x + dx, y + dy, z)]) return false;
      if (!grid.passable[grid.index(x + dx, y, z + dz)]) return false;
      if (!grid.passable[grid.index(x, y + dy, z + dz)]) return false;
    }
  }
  return true;
}

function cellSegmentBlocked(grid, a, b, margin = 0.52) {
  return segmentBlocked(grid.cellToWorld(a.x, a.y, a.z), grid.cellToWorld(b.x, b.y, b.z), margin);
}

function jpsNaturalOffsets(dx, dy, dz) {
  const values = [dx === 0 ? [0] : [0, dx], dy === 0 ? [0] : [0, dy], dz === 0 ? [0] : [0, dz]];
  const out = [];
  for (const x of values[0]) {
    for (const y of values[1]) {
      for (const z of values[2]) {
        if (x !== 0 || y !== 0 || z !== 0) out.push({ dx: x, dy: y, dz: z });
      }
    }
  }
  return out;
}

function jpsForcedOffsets(grid, x, y, z, dx, dy, dz) {
  const parent = { x: x - dx, y: y - dy, z: z - dz };
  const natural = new Set(jpsNaturalOffsets(dx, dy, dz).map((off) => `${off.dx},${off.dy},${off.dz}`));
  const forced = [];
  for (const off of neighborOffsets) {
    const key = `${off.dx},${off.dy},${off.dz}`;
    if (natural.has(key)) continue;
    if (!stepAllowed(grid, x, y, z, off.dx, off.dy, off.dz)) continue;
    const target = { x: x + off.dx, y: y + off.dy, z: z + off.dz };
    if (cellSegmentBlocked(grid, parent, target, 0.52)) forced.push(off);
  }
  return forced;
}

function jpsForced(grid, x, y, z, dx, dy, dz) {
  return jpsForcedOffsets(grid, x, y, z, dx, dy, dz).length > 0;
}

function jpsJump(grid, x, y, z, dx, dy, dz, depth = 0) {
  const limit = grid.nx + grid.ny + grid.nz;
  const order = (dx !== 0 ? 1 : 0) + (dy !== 0 ? 1 : 0) + (dz !== 0 ? 1 : 0);
  let cx = x;
  let cy = y;
  let cz = z;
  for (let step = 0; step < limit; step += 1) {
    if (!stepAllowed(grid, cx, cy, cz, dx, dy, dz)) return -1;
    cx += dx;
    cy += dy;
    cz += dz;
    const index = grid.index(cx, cy, cz);
    if (goalSet.has(index)) return index;
    if (jpsForced(grid, cx, cy, cz, dx, dy, dz)) return index;
    if (depth < 8 && order >= 2) {
      for (const natural of jpsNaturalOffsets(dx, dy, dz)) {
        const nOrder = (natural.dx !== 0 ? 1 : 0) + (natural.dy !== 0 ? 1 : 0) + (natural.dz !== 0 ? 1 : 0);
        if (nOrder >= order) continue;
        if (jpsJump(grid, cx, cy, cz, natural.dx, natural.dy, natural.dz, depth + 1) >= 0) return index;
      }
    }
  }
  return -1;
}

function jpsSuccessorOffsets(grid, index, parentIndex) {
  const c = grid.unpack(index);
  if (parentIndex < 0) return neighborOffsets.filter((off) => stepAllowed(grid, c.x, c.y, c.z, off.dx, off.dy, off.dz));
  const p = grid.unpack(parentIndex);
  const dx = Math.sign(c.x - p.x);
  const dy = Math.sign(c.y - p.y);
  const dz = Math.sign(c.z - p.z);
  const keyed = new Map();
  for (const off of jpsNaturalOffsets(dx, dy, dz)) keyed.set(`${off.dx},${off.dy},${off.dz}`, off);
  for (const off of jpsForcedOffsets(grid, c.x, c.y, c.z, dx, dy, dz)) keyed.set(`${off.dx},${off.dy},${off.dz}`, off);
  return [...keyed.values()].filter((off) => stepAllowed(grid, c.x, c.y, c.z, off.dx, off.dy, off.dz));
}

function planJumpPointSearch(start) {
  const grid = state.grid;
  if (!goalSet.size) return null;
  const gScores = new Map([[start.index, 0]]);
  const parents = new Map([[start.index, -1]]);
  const closed = new Set();
  const heap = new MinHeap();
  heap.push(start.index, distanceHeuristic(start.index));
  let found = -1;
  let expansions = 0;
  while (heap.items.length) {
    const current = heap.pop();
    if (!current) break;
    const index = current.node;
    if (closed.has(index)) continue;
    closed.add(index);
    const c = grid.unpack(index);
    if (goalSet.has(index)) {
      found = index;
      break;
    }
    if ((expansions += 1) > 14000) break;
    for (const offset of jpsSuccessorOffsets(grid, index, parents.get(index) ?? -1)) {
      const jump = jpsJump(grid, c.x, c.y, c.z, offset.dx, offset.dy, offset.dz);
      if (jump < 0 || closed.has(jump)) continue;
      const j = grid.unpack(jump);
      const cost = Math.hypot((j.x - c.x) * grid.cell, (j.y - c.y) * grid.yStep, (j.z - c.z) * grid.cell) / grid.cell;
      const ng = gScores.get(index) + cost;
      if (!gScores.has(jump) || ng < gScores.get(jump) - 1e-6) {
        gScores.set(jump, ng);
        parents.set(jump, index);
        heap.push(jump, ng + distanceHeuristic(jump));
      }
    }
  }
  if (found < 0) return null;
  const cells = [];
  let walk = found;
  while (walk >= 0) {
    const c = grid.unpack(walk);
    cells.push({ x: c.x, y: c.y, z: c.z, index: walk });
    walk = parents.get(walk) ?? -1;
  }
  cells.reverse();
  return cells;
}

function kinodynamicStepCost(currentDir, nextDir, accel) {
  const current = directionOffset(currentDir);
  const next = directionOffset(nextDir);
  const speed = Math.hypot(next.dx, next.dy * 1.25, next.dz);
  const accelMag = Math.hypot(accel.dx, accel.dy * 1.25, accel.dz);
  const jerk = Math.hypot(next.dx - current.dx, (next.dy - current.dy) * 1.25, next.dz - current.dz);
  return speed + accelMag * 0.28 + jerk * 0.18 + Math.abs(next.dy) * 0.18;
}

function planKinodynamicAStar(start) {
  const grid = state.grid;
  if (!goalSet.size) return null;
  const D = REST_DIR + 1;
  const s = ensureSearchArrays(grid.total * D);
  const heap = new MinHeap();
  const startState = start.index * D + REST_DIR;
  s.stamp[startState] = s.gen;
  s.gScore[startState] = 0;
  s.parent[startState] = -1;
  heap.push(startState, distanceHeuristic(start.index) * 0.82);
  let found = -1;
  let expansions = 0;

  while (heap.items.length) {
    const current = heap.pop();
    if (!current) break;
    const stateIndex = current.node;
    const cellIndex = Math.floor(stateIndex / D);
    const dir = stateIndex % D;
    const c = grid.unpack(cellIndex);
    const g = s.gScore[stateIndex];
    if (current.priority > g + distanceHeuristic(cellIndex) * 0.82 + 0.001) continue;
    if (goalSet.has(cellIndex)) {
      found = stateIndex;
      break;
    }
    if ((expansions += 1) > 120000) break;

    const velocity = directionOffset(dir);
    for (const accel of accelerationOffsets) {
      const vx = clamp(velocity.dx + accel.dx, -1, 1);
      const vy = clamp(velocity.dy + accel.dy, -1, 1);
      const vz = clamp(velocity.dz + accel.dz, -1, 1);
      if (vx === 0 && vy === 0 && vz === 0) continue;
      const nx = c.x + vx;
      const ny = c.y + vy;
      const nz = c.z + vz;
      if (!stepAllowed(grid, c.x, c.y, c.z, vx, vy, vz)) continue;
      const nextCell = grid.index(nx, ny, nz);
      const nextDir = directionIndexFor(vx, vy, vz);
      const nextState = nextCell * D + nextDir;
      const ng = g + kinodynamicStepCost(dir, nextDir, accel);
      if (s.stamp[nextState] !== s.gen || ng < s.gScore[nextState] - 1e-6) {
        s.stamp[nextState] = s.gen;
        s.gScore[nextState] = ng;
        s.parent[nextState] = stateIndex;
        heap.push(nextState, ng + distanceHeuristic(nextCell) * 0.82);
      }
    }
  }
  if (found < 0) return null;
  return cellsFromStateChain(found, D);
}

function primitiveCurveBlocked(grid, cell, dir, nextDir, endpoint, margin = 0.56) {
  const start = grid.cellToWorld(cell.x, cell.y, cell.z);
  const end = grid.cellToWorld(endpoint.x, endpoint.y, endpoint.z);
  const current = directionOffset(dir);
  const next = directionOffset(nextDir);
  const mid = start.clone().lerp(end, 0.5);
  mid.x += (current.dx - next.dx) * grid.cell * 0.38;
  mid.y += (current.dy - next.dy) * grid.yStep * 0.38;
  mid.z += (current.dz - next.dz) * grid.cell * 0.38;
  mid.y = clamp(mid.y, 1.2, 15);
  const point = new THREE.Vector3();
  const steps = Math.max(6, Math.ceil(start.distanceTo(end) / 0.52));
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const omt = 1 - t;
    point.copy(start).multiplyScalar(omt * omt).addScaledVector(mid, 2 * omt * t).addScaledVector(end, t * t);
    if (isWorldBlocked(point, margin)) return true;
  }
  return false;
}

function motionPrimitiveCandidates(dir) {
  const current = directionOffset(dir);
  const candidates = [];
  for (let nextDir = 0; nextDir < REST_DIR; nextDir += 1) {
    const next = directionOffset(nextDir);
    if (dir !== REST_DIR) {
      const dot = offsetUnits[dir].dot(offsetUnits[nextDir]);
      if (dot < -0.18) continue;
    }
    const dx = clamp(current.dx + next.dx, -2, 2);
    const dy = clamp(current.dy + next.dy, -1, 1);
    const dz = clamp(current.dz + next.dz, -2, 2);
    if (dx === 0 && dy === 0 && dz === 0) continue;
    candidates.push({ dx, dy, dz, nextDir });
  }
  return candidates;
}

function planMotionPrimitiveSearch(start) {
  const grid = state.grid;
  if (!goalSet.size) return null;
  const D = REST_DIR + 1;
  const s = ensureSearchArrays(grid.total * D);
  const heap = new MinHeap();
  const startState = start.index * D + REST_DIR;
  s.stamp[startState] = s.gen;
  s.gScore[startState] = 0;
  s.parent[startState] = -1;
  heap.push(startState, distanceHeuristic(start.index) * 0.78);
  let found = -1;
  let expansions = 0;

  while (heap.items.length) {
    const current = heap.pop();
    if (!current) break;
    const stateIndex = current.node;
    const cellIndex = Math.floor(stateIndex / D);
    const dir = stateIndex % D;
    const c = grid.unpack(cellIndex);
    const g = s.gScore[stateIndex];
    if (current.priority > g + distanceHeuristic(cellIndex) * 0.78 + 0.001) continue;
    if (goalSet.has(cellIndex)) {
      found = stateIndex;
      break;
    }
    if ((expansions += 1) > 90000) break;

    for (const primitive of motionPrimitiveCandidates(dir)) {
      const nx = c.x + primitive.dx;
      const ny = c.y + primitive.dy;
      const nz = c.z + primitive.dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz) continue;
      const nextCell = grid.index(nx, ny, nz);
      if (!grid.passable[nextCell]) continue;
      const endpoint = { x: nx, y: ny, z: nz, index: nextCell };
      if (primitiveCurveBlocked(grid, c, dir, primitive.nextDir, endpoint)) continue;
      const edge = grid.cellToWorld(c.x, c.y, c.z).distanceTo(grid.cellToWorld(nx, ny, nz)) / grid.cell;
      const turn = dir === REST_DIR ? 0 : 1 - clamp(offsetUnits[dir].dot(offsetUnits[primitive.nextDir]), -1, 1);
      const nextState = nextCell * D + primitive.nextDir;
      const ng = g + edge * 0.92 + turn * 0.72 + Math.abs(primitive.dy) * 0.32;
      if (s.stamp[nextState] !== s.gen || ng < s.gScore[nextState] - 1e-6) {
        s.stamp[nextState] = s.gen;
        s.gScore[nextState] = ng;
        s.parent[nextState] = stateIndex;
        heap.push(nextState, ng + distanceHeuristic(nextCell) * 0.78);
      }
    }
  }
  if (found < 0) return null;
  return cellsFromStateChain(found, D);
}

function segmentBlocked(a, b, margin = 0.5) {
  const length = a.distanceTo(b);
  const steps = Math.max(1, Math.ceil(length / 0.7));
  const point = new THREE.Vector3();
  for (let i = 0; i <= steps; i += 1) {
    point.lerpVectors(a, b, i / steps);
    if (isWorldBlocked(point, margin)) return true;
  }
  return false;
}

function goalPointFor(start) {
  const grid = state.grid;
  let best = null;
  let bestScore = Number.POSITIVE_INFINITY;
  for (const goal of state.goals) {
    const score = Math.abs(goal.z - start.z) + Math.abs(goal.y - start.y) * 1.6;
    if (score < bestScore) {
      bestScore = score;
      best = goal;
    }
  }
  return best ? grid.cellToWorld(best.x, best.y, best.z) : null;
}

function makeInformedSampler(startPoint, goalPoint, rng) {
  const cmin = startPoint.distanceTo(goalPoint);
  const mid = startPoint.clone().add(goalPoint).multiplyScalar(0.5);
  const axis = goalPoint.clone().sub(startPoint).normalize();
  const seedU = Math.abs(axis.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const basisU = seedU.cross(axis).normalize();
  const basisV = axis.clone().cross(basisU).normalize();
  return (bestCost, out) => {
    if (!Number.isFinite(bestCost)) {
      out.set(rng() * 66 - 33, 1.3 + rng() * 13.2, rng() * 66 - 33);
    } else {
      let x;
      let y;
      let z;
      let lenSq;
      do {
        x = rng() * 2 - 1;
        y = rng() * 2 - 1;
        z = rng() * 2 - 1;
        lenSq = x * x + y * y + z * z;
      } while (lenSq > 1 || lenSq < 1e-6);
      const cBest = Math.max(bestCost, cmin + 0.01);
      const major = cBest / 2;
      const minor = Math.sqrt(Math.max(0.01, cBest * cBest - cmin * cmin)) / 2;
      out.copy(mid).addScaledVector(axis, x * major).addScaledVector(basisU, y * minor).addScaledVector(basisV, z * minor);
    }
    out.y = clamp(out.y, 1.3, 14.6);
    out.x = clamp(out.x, -34, 34);
    out.z = clamp(out.z, -34, 34);
    return out;
  };
}

function sampleRrtWorkspace(startPoint, goalPoint, rng, out) {
  if (rng() < 0.34) {
    out.copy(startPoint).lerp(goalPoint, rng());
    const spread = 5.2 + startPoint.distanceTo(goalPoint) * 0.04;
    out.x += (rng() - 0.5) * spread;
    out.y += (rng() - 0.5) * Math.min(7.5, spread * 0.85);
    out.z += (rng() - 0.5) * spread;
  } else {
    out.set(rng() * 66 - 33, 1.3 + rng() * 13.2, rng() * 66 - 33);
  }
  out.x = clamp(out.x, -34, 34);
  out.y = clamp(out.y, 1.3, 14.6);
  out.z = clamp(out.z, -34, 34);
  return out;
}

function sampleFreeRrtPoint(startPoint, goalPoint, rng, informedSampler, bestCost, useInformed, out) {
  for (let attempt = 0; attempt < 16; attempt += 1) {
    if (rng() < 0.075) out.copy(goalPoint);
    else if (useInformed && Number.isFinite(bestCost)) informedSampler(bestCost, out);
    else sampleRrtWorkspace(startPoint, goalPoint, rng, out);
    if (Number.isFinite(bestCost) && startPoint.distanceTo(out) + out.distanceTo(goalPoint) >= bestCost - 0.03) continue;
    if (!isWorldBlocked(out, 0.55)) return true;
  }
  return false;
}

function rrtConnectionRadius(count, step) {
  const n = Math.max(2, count);
  const radius = 18.5 * Math.cbrt(Math.log(n + 1) / n);
  return clamp(radius, step * 1.18, 7.2);
}

function addTreeNode(nodes, point, parent, cost) {
  const index = nodes.length;
  nodes.push({ point: point.clone(), parent, cost, children: [] });
  if (parent >= 0) nodes[parent].children.push(index);
  return index;
}

function removeTreeChild(nodes, parent, child) {
  if (parent < 0) return;
  const children = nodes[parent].children;
  const at = children.indexOf(child);
  if (at >= 0) children.splice(at, 1);
}

function propagateTreeCost(nodes, root, delta) {
  const stack = [...nodes[root].children];
  while (stack.length) {
    const index = stack.pop();
    nodes[index].cost += delta;
    stack.push(...nodes[index].children);
  }
}

function rewireTreeNode(nodes, index, parent, cost) {
  const node = nodes[index];
  if (node.parent === parent) return;
  removeTreeChild(nodes, node.parent, index);
  nodes[parent].children.push(index);
  const delta = cost - node.cost;
  node.parent = parent;
  node.cost = cost;
  propagateTreeCost(nodes, index, delta);
}

function nearestTreeNode(nodes, point) {
  let best = 0;
  let bestSq = Number.POSITIVE_INFINITY;
  for (let i = 0; i < nodes.length; i += 1) {
    const dSq = nodes[i].point.distanceToSquared(point);
    if (dSq < bestSq) {
      bestSq = dSq;
      best = i;
    }
  }
  return best;
}

function treePath(nodes, index, goalPoint = null) {
  const points = [];
  if (goalPoint) points.push(goalPoint.clone());
  let walk = index;
  let guard = nodes.length + 4;
  while (walk >= 0 && guard-- > 0) {
    points.push(nodes[walk].point.clone());
    walk = nodes[walk].parent;
  }
  points.reverse();
  return points;
}

function planRrtStar(start, informed) {
  const grid = state.grid;
  const startPoint = grid.cellToWorld(start.x, start.y, start.z);
  const goalPoint = goalPointFor(start);
  if (!goalPoint) return null;
  const rng = createRng((scenarioSeed(state.scenario) ^ Math.imul(start.index, 2654435761)) + (informed ? 977 : 331));
  const sampler = makeInformedSampler(startPoint, goalPoint, rng);
  const nodes = [{ point: startPoint.clone(), parent: -1, cost: 0, children: [] }];
  const sample = new THREE.Vector3();
  const step = informed ? 3.35 : 3.15;
  const iterations = informed ? 760 : 620;
  let bestGoal = -1;
  let bestCost = Number.POSITIVE_INFINITY;

  for (let it = 0; it < iterations; it += 1) {
    if (bestGoal >= 0) bestCost = nodes[bestGoal].cost + nodes[bestGoal].point.distanceTo(goalPoint);
    if (!sampleFreeRrtPoint(startPoint, goalPoint, rng, sampler, bestCost, informed, sample)) continue;

    const nearestIndex = nearestTreeNode(nodes, sample);
    const nearNode = nodes[nearestIndex];
    const toward = sample.clone().sub(nearNode.point);
    const distance = toward.length();
    if (distance < 0.08) continue;
    if (distance > step) toward.multiplyScalar(step / distance);
    const newPoint = nearNode.point.clone().add(toward);
    if (isWorldBlocked(newPoint, 0.55) || segmentBlocked(nearNode.point, newPoint, 0.55)) continue;

    const radius = rrtConnectionRadius(nodes.length + 1, step);
    const radiusSq = radius * radius;
    const nearIndexes = [];
    let parent = nearestIndex;
    let cost = nearNode.cost + nearNode.point.distanceTo(newPoint);

    for (let i = 0; i < nodes.length; i += 1) {
      const dSq = nodes[i].point.distanceToSquared(newPoint);
      if (dSq <= radiusSq) {
        nearIndexes.push(i);
        const candidateCost = nodes[i].cost + Math.sqrt(dSq);
        if (candidateCost < cost - 0.01 && candidateCost + newPoint.distanceTo(goalPoint) < bestCost - 0.01 && !segmentBlocked(nodes[i].point, newPoint, 0.55)) {
          parent = i;
          cost = candidateCost;
        }
      }
    }

    const nodeIndex = addTreeNode(nodes, newPoint, parent, cost);

    for (const i of nearIndexes) {
      if (i === parent || i === 0) continue;
      const edgeCost = newPoint.distanceTo(nodes[i].point);
      const through = cost + edgeCost;
      if (through + 0.01 < nodes[i].cost && through + nodes[i].point.distanceTo(goalPoint) < bestCost + 0.25 && !segmentBlocked(newPoint, nodes[i].point, 0.55)) {
        rewireTreeNode(nodes, i, nodeIndex, through);
      }
    }

    const goalDistance = newPoint.distanceTo(goalPoint);
    const connectRadius = Math.max(step * 1.18, radius * 0.72);
    if (goalDistance <= connectRadius && !segmentBlocked(newPoint, goalPoint, 0.55)) {
      const total = nodes[nodeIndex].cost + goalDistance;
      if (total < bestCost - 0.01) {
        bestCost = total;
        bestGoal = nodeIndex;
      }
    }
  }

  if (bestGoal < 0) return null;
  return shortcutPoints(treePath(nodes, bestGoal, goalPoint), 0.5);
}

function addBitSample(samples, point) {
  samples.points.push(point.clone());
  samples.g.push(Number.POSITIVE_INFINITY);
  samples.parent.push(-1);
  samples.children.push([]);
  samples.inTree.push(false);
  return samples.points.length - 1;
}

function addBitVertex(samples, index, parent, cost) {
  samples.inTree[index] = true;
  samples.parent[index] = parent;
  samples.g[index] = cost;
  if (parent >= 0) samples.children[parent].push(index);
}

function removeBitChild(samples, parent, child) {
  if (parent < 0) return;
  const children = samples.children[parent];
  const at = children.indexOf(child);
  if (at >= 0) children.splice(at, 1);
}

function propagateBitCost(samples, root, delta) {
  const stack = [...samples.children[root]];
  while (stack.length) {
    const index = stack.pop();
    samples.g[index] += delta;
    stack.push(...samples.children[index]);
  }
}

function rewireBitVertex(samples, index, parent, cost) {
  removeBitChild(samples, samples.parent[index], index);
  samples.parent[index] = parent;
  samples.children[parent].push(index);
  const delta = cost - samples.g[index];
  samples.g[index] = cost;
  propagateBitCost(samples, index, delta);
}

function bitConnectionRadius(count) {
  const n = Math.max(2, count);
  return clamp(25 * Math.cbrt(Math.log(n + 1) / n), 4.8, 8.8);
}

function bitPath(samples, goalIndex) {
  const points = [];
  let walk = goalIndex;
  let guard = samples.points.length + 4;
  while (walk >= 0 && guard-- > 0) {
    points.push(samples.points[walk].clone());
    walk = samples.parent[walk];
  }
  points.reverse();
  return shortcutPoints(points, 0.5);
}

function planBitStar(start) {
  const grid = state.grid;
  const startPoint = grid.cellToWorld(start.x, start.y, start.z);
  const goalPoint = goalPointFor(start);
  if (!goalPoint) return null;
  const rng = createRng((scenarioSeed(state.scenario) ^ Math.imul(start.index, 40503)) + 613);
  const sampler = makeInformedSampler(startPoint, goalPoint, rng);
  const samples = {
    points: [startPoint.clone(), goalPoint.clone()],
    g: [0, Number.POSITIVE_INFINITY],
    parent: [-1, -1],
    children: [[], []],
    inTree: [true, false],
  };
  const sample = new THREE.Vector3();
  const edgeCache = new Map();
  const edgeFree = (i, j) => {
    const key = i < j ? `${i}:${j}` : `${j}:${i}`;
    let free = edgeCache.get(key);
    if (free === undefined) {
      free = !segmentBlocked(samples.points[i], samples.points[j], 0.55);
      edgeCache.set(key, free);
    }
    return free;
  };
  let bestCost = Number.POSITIVE_INFINITY;
  let bestGoal = -1;

  const seedRoute = tracePathFromCell(start);
  const seeded = densifyPolyline(seedRoute.smooth, 3.2);
  for (let i = 1; i < seeded.length - 1; i += 1) if (!isWorldBlocked(seeded[i], 0.55)) addBitSample(samples, seeded[i]);

  for (let batch = 0; batch < 4; batch += 1) {
    const batchSamples = batch === 0 ? 90 : 105;
    for (let sIdx = 0; sIdx < batchSamples; sIdx += 1) {
      if (!sampleFreeRrtPoint(startPoint, goalPoint, rng, sampler, bestCost, Number.isFinite(bestCost), sample)) continue;
      addBitSample(samples, sample);
    }

    const radius = bitConnectionRadius(samples.points.length);
    const radiusSq = radius * radius;
    const vertexQueue = new MinHeap();
    const edgeQueue = new MinHeap();
    const expanded = new Set();
    for (let i = 0; i < samples.points.length; i += 1) {
      if (samples.inTree[i] && samples.g[i] + samples.points[i].distanceTo(goalPoint) < bestCost - 0.01) {
        vertexQueue.push(i, samples.g[i] + samples.points[i].distanceTo(goalPoint));
      }
    }

    let processed = 0;
    const maxEdges = batch === 0 ? 2200 : 1800;
    while ((vertexQueue.items.length || edgeQueue.items.length) && processed < maxEdges) {
      while (vertexQueue.items.length && (!edgeQueue.items.length || vertexQueue.items[0].priority <= edgeQueue.items[0].priority)) {
        const item = vertexQueue.pop();
        if (!item || expanded.has(item.node)) continue;
        const v = item.node;
        expanded.add(v);
        const pv = samples.points[v];
        for (let x = 1; x < samples.points.length; x += 1) {
          if (x === v) continue;
          const dSq = pv.distanceToSquared(samples.points[x]);
          if (dSq > radiusSq) continue;
          const edgeLower = Math.sqrt(dSq);
          const lowerBound = samples.g[v] + edgeLower + samples.points[x].distanceTo(goalPoint);
          if (lowerBound >= bestCost - 0.01) continue;
          if (samples.inTree[x] && samples.g[v] + edgeLower >= samples.g[x] - 0.01) continue;
          edgeQueue.push({ v, x, edgeLower }, lowerBound);
        }
      }

      const edge = edgeQueue.pop();
      if (!edge) break;
      processed += 1;
      const { v, x, edgeLower } = edge.node;
      const candidateCost = samples.g[v] + edgeLower;
      if (candidateCost + samples.points[x].distanceTo(goalPoint) >= bestCost - 0.01) continue;
      if (candidateCost >= samples.g[x] - 0.01) continue;
      if (!edgeFree(v, x)) continue;

      if (!samples.inTree[x]) addBitVertex(samples, x, v, candidateCost);
      else rewireBitVertex(samples, x, v, candidateCost);

      if (x === 1) {
        bestCost = candidateCost;
        bestGoal = 1;
      } else if (candidateCost + samples.points[x].distanceTo(goalPoint) < bestCost - 0.01) {
        vertexQueue.push(x, candidateCost + samples.points[x].distanceTo(goalPoint));
      }
    }
  }

  return bestGoal >= 0 ? bitPath(samples, bestGoal) : null;
}

function shortcutPoints(points, margin = 0.5) {
  if (points.length <= 2) return points;
  const out = [points[0]];
  let i = 0;
  while (i < points.length - 1) {
    let j = points.length - 1;
    for (; j > i + 1; j -= 1) {
      if (!segmentBlocked(points[i], points[j], margin)) break;
    }
    out.push(points[j]);
    i = j;
  }
  return out;
}

function routeFromCells(cells) {
  const grid = state.grid;
  const raw = cells.map((cell) => grid.cellToWorld(cell.x, cell.y, cell.z));
  const simplified = simplifyCells(cells).map((cell) => grid.cellToWorld(cell.x, cell.y, cell.z));
  const smooth = smoothPath(simplified);
  return { cells, raw, smooth: smooth.length > 1 ? smooth : raw };
}

function densifyPolyline(points, spacing = 1.2) {
  const out = [points[0].clone()];
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const steps = Math.max(1, Math.ceil(a.distanceTo(b) / spacing));
    for (let s = 1; s <= steps; s += 1) out.push(a.clone().lerp(b, s / steps));
  }
  return out;
}

function routeFromPoints(points) {
  const raw = points.map((point) => point.clone());
  const sparse = shortcutPoints(points);
  const smooth = smoothPath(densifyPolyline(sparse));
  return { cells: null, raw, smooth: smooth.length > 1 ? smooth : raw };
}

// --- B-family trajectory optimizers ---

const STENCIL_ACCEL = [1, -2, 1];
const STENCIL_JERK = [-1, 3, -3, 1];
const STENCIL_SNAP = [1, -4, 6, -4, 1];

function refinePath(points, opts) {
  const pts = points.map((point) => point.clone());
  const n = pts.length;
  if (n < 6) return pts;
  const stencil = opts.stencil;
  const step = opts.step ?? 0.02;
  const orig = opts.tube ? points.map((point) => point.clone()) : null;
  const grad = pts.map(() => new THREE.Vector3());
  const tmp = new THREE.Vector3();
  for (let iter = 0; iter < (opts.iterations ?? 24); iter += 1) {
    for (const g of grad) g.set(0, 0, 0);
    for (let j = 0; j + stencil.length <= n; j += 1) {
      tmp.set(0, 0, 0);
      for (let m = 0; m < stencil.length; m += 1) tmp.addScaledVector(pts[j + m], stencil[m]);
      for (let m = 0; m < stencil.length; m += 1) grad[j + m].addScaledVector(tmp, stencil[m]);
    }
    for (let i = 2; i < n - 2; i += 1) {
      const move = grad[i].multiplyScalar(-step);
      const len = move.length();
      if (len > 0.25) move.multiplyScalar(0.25 / len);
      pts[i].add(move);
      if (opts.esdfMargin) {
        const d = sampleEsdf(pts[i]);
        if (d < opts.esdfMargin) {
          esdfGradient(pts[i], tmp);
          pts[i].addScaledVector(tmp, (opts.esdfMargin - d) * (opts.esdfWeight ?? 0.3));
        }
      }
      if (orig) {
        tmp.copy(pts[i]).sub(orig[i]);
        const dist = tmp.length();
        if (dist > opts.tube) pts[i].addScaledVector(tmp.normalize(), -(dist - opts.tube));
      }
      pts[i].y = clamp(pts[i].y, 1.2, 15);
    }
  }
  return pts;
}

function uniqueTrajectoryPoints(points) {
  const out = [];
  for (const point of points) {
    if (!out.length || out[out.length - 1].distanceToSquared(point) > 0.04) out.push(point.clone());
  }
  return out.length > 1 ? out : points.map((point) => point.clone());
}

function sparseTrajectoryWaypoints(route) {
  const raw = route.raw?.length ? route.raw : route.smooth;
  let sparse = shortcutPoints(raw).map((point) => point.clone());
  if (sparse.length < 3) sparse = uniqueTrajectoryPoints(route.smooth);
  if (sparse.length > 22) {
    const reduced = [sparse[0].clone()];
    const stride = Math.ceil((sparse.length - 2) / 20);
    for (let i = stride; i < sparse.length - 1; i += stride) reduced.push(sparse[i].clone());
    reduced.push(sparse[sparse.length - 1].clone());
    sparse = reduced;
  }
  return uniqueTrajectoryPoints(sparse);
}

function segmentDurations(points, nominalSpeed = 3.1) {
  const durations = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    durations.push(clamp(points[i].distanceTo(points[i + 1]) / nominalSpeed, 0.42, 2.8));
  }
  return durations;
}

function waypointDerivatives(points, durations, includeJerk) {
  const n = points.length;
  const velocity = points.map(() => new THREE.Vector3());
  const accel = points.map(() => new THREE.Vector3());
  const jerk = points.map(() => new THREE.Vector3());

  for (let i = 1; i < n - 1; i += 1) {
    const span = durations[i - 1] + durations[i];
    velocity[i].copy(points[i + 1]).sub(points[i - 1]).multiplyScalar(1 / Math.max(span, 0.001));
    const dt = Math.max(0.5 * span, 0.001);
    accel[i]
      .copy(points[i + 1])
      .add(points[i - 1])
      .addScaledVector(points[i], -2)
      .multiplyScalar(1 / (dt * dt));
  }

  if (includeJerk) {
    for (let i = 2; i < n - 2; i += 1) {
      const dt = Math.max((durations[i - 2] + durations[i - 1] + durations[i] + durations[i + 1]) / 4, 0.001);
      jerk[i]
        .copy(points[i + 2])
        .addScaledVector(points[i + 1], -2)
        .addScaledVector(points[i - 1], 2)
        .addScaledVector(points[i - 2], -1)
        .multiplyScalar(1 / (2 * dt * dt * dt));
    }
  }

  return { velocity, accel, jerk };
}

function quinticHermiteCoefficients(p0, p1, v0, v1, a0, a1, T) {
  const t2 = T * T;
  const t3 = t2 * T;
  const t4 = t3 * T;
  const t5 = t4 * T;
  return [
    p0,
    v0,
    a0 / 2,
    (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * t2) / (2 * t3),
    (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * t2) / (2 * t4),
    (12 * (p1 - p0) - (6 * v1 + 6 * v0) * T - (a0 - a1) * t2) / (2 * t5),
  ];
}

function solveLinear4(matrix, rhs) {
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < 4; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 4; row += 1) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (pivot !== col) [a[pivot], a[col]] = [a[col], a[pivot]];
    const div = Math.abs(a[col][col]) < 1e-9 ? 1e-9 : a[col][col];
    for (let j = col; j <= 4; j += 1) a[col][j] /= div;
    for (let row = 0; row < 4; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= 4; j += 1) a[row][j] -= factor * a[col][j];
    }
  }
  return [a[0][4], a[1][4], a[2][4], a[3][4]];
}

function septicHermiteCoefficients(p0, p1, v0, v1, a0, a1, j0, j1, T) {
  const t2 = T * T;
  const t3 = t2 * T;
  const t4 = t3 * T;
  const t5 = t4 * T;
  const t6 = t5 * T;
  const t7 = t6 * T;
  const c0 = p0;
  const c1 = v0;
  const c2 = a0 / 2;
  const c3 = j0 / 6;
  const rhs = [
    p1 - (c0 + c1 * T + c2 * t2 + c3 * t3),
    v1 - (c1 + 2 * c2 * T + 3 * c3 * t2),
    a1 - (2 * c2 + 6 * c3 * T),
    j1 - 6 * c3,
  ];
  const hi = solveLinear4(
    [
      [t4, t5, t6, t7],
      [4 * t3, 5 * t4, 6 * t5, 7 * t6],
      [12 * t2, 20 * t3, 30 * t4, 42 * t5],
      [24 * T, 60 * t2, 120 * t3, 210 * t4],
    ],
    rhs,
  );
  return [c0, c1, c2, c3, ...hi];
}

function evalScalarPolynomial(coeffs, t) {
  let value = 0;
  for (let i = coeffs.length - 1; i >= 0; i -= 1) value = value * t + coeffs[i];
  return value;
}

function samplePolynomialSegment(coeffs, T, samples, includeStart) {
  const out = [];
  const start = includeStart ? 0 : 1;
  for (let s = start; s <= samples; s += 1) {
    const t = (T * s) / samples;
    out.push(
      new THREE.Vector3(
        evalScalarPolynomial(coeffs.x, t),
        evalScalarPolynomial(coeffs.y, t),
        evalScalarPolynomial(coeffs.z, t),
      ),
    );
  }
  return out;
}

function optimizePolynomialTrajectory(points, order) {
  const waypoints = uniqueTrajectoryPoints(points);
  if (waypoints.length < 2) return waypoints;
  const durations = segmentDurations(waypoints, order === "snap" ? 3.25 : 3.05);
  const derivatives = waypointDerivatives(waypoints, durations, order === "snap");
  const out = [];

  for (let i = 0; i < waypoints.length - 1; i += 1) {
    const T = durations[i];
    const coeffs = { x: null, y: null, z: null };
    for (const axis of ["x", "y", "z"]) {
      coeffs[axis] =
        order === "snap"
          ? septicHermiteCoefficients(
              waypoints[i][axis],
              waypoints[i + 1][axis],
              derivatives.velocity[i][axis],
              derivatives.velocity[i + 1][axis],
              derivatives.accel[i][axis],
              derivatives.accel[i + 1][axis],
              derivatives.jerk[i][axis],
              derivatives.jerk[i + 1][axis],
              T,
            )
          : quinticHermiteCoefficients(
              waypoints[i][axis],
              waypoints[i + 1][axis],
              derivatives.velocity[i][axis],
              derivatives.velocity[i + 1][axis],
              derivatives.accel[i][axis],
              derivatives.accel[i + 1][axis],
              T,
            );
    }
    const samples = Math.max(4, Math.ceil(waypoints[i].distanceTo(waypoints[i + 1]) / 0.85));
    out.push(...samplePolynomialSegment(coeffs, T, samples, i === 0));
  }

  return out;
}

function enforceTrajectoryClearance(points, margin, passes = 2) {
  const out = points.map((point) => point.clone());
  const grad = new THREE.Vector3();
  for (let pass = 0; pass < passes; pass += 1) {
    for (let i = 1; i < out.length - 1; i += 1) {
      const clearance = sampleEsdf(out[i]);
      if (clearance < margin) {
        esdfGradient(out[i], grad);
        out[i].addScaledVector(grad, (margin - clearance) * 0.62);
      }
      out[i].y = clamp(out[i].y, 1.2, 15);
    }
  }
  return out;
}

function optimizeMinimumJerkTrajectory(points) {
  const trajectory = optimizePolynomialTrajectory(points, "jerk");
  return enforceTrajectoryClearance(trajectory, 1.18, 2);
}

function septicSnapEnergy(p0, p1, v0, v1, a0, a1, j0, j1, T) {
  // Closed-form integral of snap^2 over one septic segment: snap(t) = A + B t + C t^2 + D t^3.
  const c = septicHermiteCoefficients(p0, p1, v0, v1, a0, a1, j0, j1, T);
  const A = 24 * c[4];
  const B = 120 * c[5];
  const C = 360 * c[6];
  const D = 840 * c[7];
  const T2 = T * T;
  const T3 = T2 * T;
  const T4 = T3 * T;
  const T5 = T4 * T;
  const T6 = T5 * T;
  const T7 = T6 * T;
  return (
    A * A * T +
    A * B * T2 +
    ((B * B + 2 * A * C) * T3) / 3 +
    ((2 * A * D + 2 * B * C) * T4) / 4 +
    ((C * C + 2 * B * D) * T5) / 5 +
    (2 * C * D * T6) / 6 +
    (D * D * T7) / 7
  );
}

function solveSymmetric3(m, b) {
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-9) return null;
  const inv = 1 / det;
  const c0 =
    b[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (b[1] * m[2][2] - m[1][2] * b[2]) +
    m[0][2] * (b[1] * m[2][1] - m[1][1] * b[2]);
  const c1 =
    m[0][0] * (b[1] * m[2][2] - m[1][2] * b[2]) -
    b[0] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * b[2] - b[1] * m[2][0]);
  const c2 =
    m[0][0] * (m[1][1] * b[2] - b[1] * m[2][1]) -
    m[0][1] * (m[1][0] * b[2] - b[1] * m[2][0]) +
    b[0] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  return [c0 * inv, c1 * inv, c2 * inv];
}

function minimizeQuadratic3(f, x0, y0, z0) {
  // f is exactly quadratic in its 3 arguments, so a single finite-difference Newton
  // step lands on the exact minimiser (Hessian is constant).
  const h = 1;
  const f0 = f(x0, y0, z0);
  const fx = f(x0 + h, y0, z0);
  const fxm = f(x0 - h, y0, z0);
  const fy = f(x0, y0 + h, z0);
  const fym = f(x0, y0 - h, z0);
  const fz = f(x0, y0, z0 + h);
  const fzm = f(x0, y0, z0 - h);
  const g = [(fx - fxm) / (2 * h), (fy - fym) / (2 * h), (fz - fzm) / (2 * h)];
  const Hxx = (fx - 2 * f0 + fxm) / (h * h);
  const Hyy = (fy - 2 * f0 + fym) / (h * h);
  const Hzz = (fz - 2 * f0 + fzm) / (h * h);
  const Hxy =
    (f(x0 + h, y0 + h, z0) - f(x0 + h, y0 - h, z0) - f(x0 - h, y0 + h, z0) + f(x0 - h, y0 - h, z0)) / (4 * h * h);
  const Hxz =
    (f(x0 + h, y0, z0 + h) - f(x0 + h, y0, z0 - h) - f(x0 - h, y0, z0 + h) + f(x0 - h, y0, z0 - h)) / (4 * h * h);
  const Hyz =
    (f(x0, y0 + h, z0 + h) - f(x0, y0 + h, z0 - h) - f(x0, y0 - h, z0 + h) + f(x0, y0 - h, z0 - h)) / (4 * h * h);
  const delta = solveSymmetric3(
    [
      [Hxx, Hxy, Hxz],
      [Hxy, Hyy, Hyz],
      [Hxz, Hyz, Hzz],
    ],
    [-g[0], -g[1], -g[2]],
  );
  if (!delta) return [x0, y0, z0];
  return [x0 + delta[0], y0 + delta[1], z0 + delta[2]];
}

function optimizeGlobalMinSnap(points) {
  // Global minimum-snap QP solved by block coordinate descent: the free variables are
  // the velocity/accel/jerk at each interior waypoint, the objective is the total
  // integral of snap^2, and each block solve is the exact minimiser of the two adjacent
  // segments' snap energy. Axes are independent; ends are rest-to-rest.
  const waypoints = uniqueTrajectoryPoints(points);
  const n = waypoints.length;
  if (n < 3) return optimizePolynomialTrajectory(waypoints, "snap");
  const durations = segmentDurations(waypoints, 3.25);
  const init = waypointDerivatives(waypoints, durations, true);
  const vel = waypoints.map((_, i) => init.velocity[i].clone());
  const acc = waypoints.map((_, i) => init.accel[i].clone());
  const jerk = waypoints.map((_, i) => init.jerk[i].clone());
  for (const i of [0, n - 1]) {
    vel[i].set(0, 0, 0);
    acc[i].set(0, 0, 0);
    jerk[i].set(0, 0, 0);
  }

  const axes = ["x", "y", "z"];
  const localCost = (axis, i, v, a, j) => {
    let e = 0;
    if (i > 0) {
      const T = durations[i - 1];
      e += septicSnapEnergy(
        waypoints[i - 1][axis], waypoints[i][axis],
        vel[i - 1][axis], v, acc[i - 1][axis], a, jerk[i - 1][axis], j, T,
      );
    }
    if (i < n - 1) {
      const T = durations[i];
      e += septicSnapEnergy(
        waypoints[i][axis], waypoints[i + 1][axis],
        v, vel[i + 1][axis], a, acc[i + 1][axis], j, jerk[i + 1][axis], T,
      );
    }
    return e;
  };

  for (let sweep = 0; sweep < 8; sweep += 1) {
    for (let i = 1; i < n - 1; i += 1) {
      for (const axis of axes) {
        const solved = minimizeQuadratic3(
          (v, a, j) => localCost(axis, i, v, a, j),
          vel[i][axis], acc[i][axis], jerk[i][axis],
        );
        vel[i][axis] = solved[0];
        acc[i][axis] = solved[1];
        jerk[i][axis] = solved[2];
      }
    }
  }

  const out = [];
  for (let i = 0; i < n - 1; i += 1) {
    const T = durations[i];
    const coeffs = { x: null, y: null, z: null };
    for (const axis of axes) {
      coeffs[axis] = septicHermiteCoefficients(
        waypoints[i][axis], waypoints[i + 1][axis],
        vel[i][axis], vel[i + 1][axis],
        acc[i][axis], acc[i + 1][axis],
        jerk[i][axis], jerk[i + 1][axis],
        T,
      );
    }
    const samples = Math.max(4, Math.ceil(waypoints[i].distanceTo(waypoints[i + 1]) / 0.85));
    out.push(...samplePolynomialSegment(coeffs, T, samples, i === 0));
  }
  return out;
}

function optimizeMinimumSnapTrajectory(points) {
  let trajectory = optimizeGlobalMinSnap(points);
  const bad = trajectory.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z));
  if (bad || trajectory.length < 2) trajectory = optimizePolynomialTrajectory(points, "snap");
  return enforceTrajectoryClearance(trajectory, 1.2, 2);
}

function buildSafeFlightCorridor(reference, baseRadius) {
  return reference.map((center) => {
    const clearance = sampleEsdf(center);
    const radius = clamp(Math.min(baseRadius, clearance * 0.72), 0.62, baseRadius);
    return {
      center: center.clone(),
      half: new THREE.Vector3(radius * 1.25, radius * 0.95, radius * 1.25),
    };
  });
}

function projectToCorridor(point, box) {
  point.x = clamp(point.x, box.center.x - box.half.x, box.center.x + box.half.x);
  point.y = clamp(point.y, Math.max(1.2, box.center.y - box.half.y), Math.min(15, box.center.y + box.half.y));
  point.z = clamp(point.z, box.center.z - box.half.z, box.center.z + box.half.z);
}

function optimizeSafeCorridorTrajectory(points, radius = 1.55) {
  const reference = densifyPolyline(uniqueTrajectoryPoints(points), 0.9);
  if (reference.length < 5) return reference;
  const corridor = buildSafeFlightCorridor(reference, radius);
  const pts = reference.map((point) => point.clone());
  const grad = pts.map(() => new THREE.Vector3());
  const tmp = new THREE.Vector3();

  for (let iter = 0; iter < 42; iter += 1) {
    for (const g of grad) g.set(0, 0, 0);
    for (let i = 1; i < pts.length - 1; i += 1) {
      tmp.copy(pts[i]).multiplyScalar(2).sub(pts[i - 1]).sub(pts[i + 1]);
      grad[i].addScaledVector(tmp, 0.74);
      grad[i].addScaledVector(pts[i].clone().sub(reference[i]), 0.18);
      const clearance = sampleEsdf(pts[i]);
      if (clearance < 1.0) {
        esdfGradient(pts[i], tmp);
        grad[i].addScaledVector(tmp, -(1.0 - clearance) * 1.45);
      }
    }
    for (let i = 1; i < pts.length - 1; i += 1) {
      pts[i].addScaledVector(grad[i], -0.105);
      projectToCorridor(pts[i], corridor[i]);
    }
  }

  return pts;
}
function bsplinePoint(a, b, c, d, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  const w0 = (1 - 3 * t + 3 * t2 - t3) / 6;
  const w1 = (4 - 6 * t2 + 3 * t3) / 6;
  const w2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6;
  const w3 = t3 / 6;
  return new THREE.Vector3(
    a.x * w0 + b.x * w1 + c.x * w2 + d.x * w3,
    a.y * w0 + b.y * w1 + c.y * w2 + d.y * w3,
    a.z * w0 + b.z * w1 + c.z * w2 + d.z * w3,
  );
}

function clampTrajectoryPoint(point) {
  point.x = clamp(point.x, -34.8, 34.8);
  point.y = clamp(point.y, 1.2, 15);
  point.z = clamp(point.z, -34.8, 34.8);
  return point;
}

function bsplineControlPolygon(points, spacing = 2.3) {
  const reference = densifyPolyline(uniqueTrajectoryPoints(points), Math.max(0.7, spacing * 0.45));
  if (reference.length <= 2) return reference.map((point) => point.clone());
  const ctrl = [reference[0].clone()];
  let since = 0;
  for (let i = 1; i < reference.length - 1; i += 1) {
    since += reference[i].distanceTo(reference[i - 1]);
    if (since >= spacing) {
      ctrl.push(reference[i].clone());
      since = 0;
    }
  }
  ctrl.push(reference[reference.length - 1].clone());
  while (ctrl.length < 4) {
    const a = ctrl[ctrl.length - 2] ?? ctrl[0];
    const b = ctrl[ctrl.length - 1];
    ctrl.splice(ctrl.length - 1, 0, a.clone().lerp(b, 0.5));
  }
  return ctrl;
}

function clampedCubicControls(control) {
  const first = control[0].clone();
  const last = control[control.length - 1].clone();
  return [first.clone(), first.clone(), ...control.map((point) => point.clone()), last.clone(), last.clone()];
}

function synchronizeBsplineEndpoints(ctrl) {
  ctrl[0].copy(ctrl[2]);
  ctrl[1].copy(ctrl[2]);
  ctrl[ctrl.length - 1].copy(ctrl[ctrl.length - 3]);
  ctrl[ctrl.length - 2].copy(ctrl[ctrl.length - 3]);
}

function optimizeBsplineControls(control, opts) {
  const ctrl = clampedCubicControls(control);
  const guide = ctrl.map((point) => point.clone());
  const grad = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const neighbor = new THREE.Vector3();
  const iterations = opts.iterations ?? 48;
  const step = opts.step ?? 0.055;
  const smoothWeight = opts.smoothWeight ?? 0.92;
  const guideWeight = opts.guideWeight ?? 0.1;
  const obstacleMargin = opts.obstacleMargin ?? 1.55;
  const obstacleWeight = opts.obstacleWeight ?? 2.0;
  const feasibilityWeight = opts.feasibilityWeight ?? 0.24;
  const maxControlSpacing = opts.maxControlSpacing ?? 3.0;

  for (let iter = 0; iter < iterations; iter += 1) {
    synchronizeBsplineEndpoints(ctrl);
    for (let i = 3; i < ctrl.length - 3; i += 1) {
      grad.set(0, 0, 0);
      tmp.copy(ctrl[i]).multiplyScalar(2).sub(ctrl[i - 1]).sub(ctrl[i + 1]);
      grad.addScaledVector(tmp, smoothWeight);
      grad.addScaledVector(ctrl[i].clone().sub(guide[i]), guideWeight);

      for (const j of [i - 1, i + 1]) {
        neighbor.copy(ctrl[i]).sub(ctrl[j]);
        const distance = neighbor.length();
        if (distance > maxControlSpacing) {
          grad.addScaledVector(neighbor, ((distance - maxControlSpacing) / distance) * feasibilityWeight);
        }
      }

      const clearance = sampleEsdf(ctrl[i]);
      if (clearance < obstacleMargin) {
        esdfGradient(ctrl[i], tmp);
        grad.addScaledVector(tmp, -(obstacleMargin - clearance) * obstacleWeight);
      }

      const move = grad.multiplyScalar(-step);
      if (move.length() > 0.42) move.setLength(0.42);
      ctrl[i].add(move);
      clampTrajectoryPoint(ctrl[i]);
    }
  }
  synchronizeBsplineEndpoints(ctrl);
  return ctrl;
}

function sampleBsplineControls(ctrl, samplesPerSpan = 6) {
  const out = [];
  for (let j = 0; j + 3 < ctrl.length; j += 1) {
    for (let s = 0; s < samplesPerSpan; s += 1) {
      const point = bsplinePoint(ctrl[j], ctrl[j + 1], ctrl[j + 2], ctrl[j + 3], s / samplesPerSpan);
      if (!out.length || out[out.length - 1].distanceToSquared(point) > 0.01) out.push(point);
    }
  }
  out.push(bsplinePoint(ctrl[ctrl.length - 4], ctrl[ctrl.length - 3], ctrl[ctrl.length - 2], ctrl[ctrl.length - 1], 1));
  return out;
}

function optimizeBspline(points, opts = {}) {
  const src = uniqueTrajectoryPoints(points);
  if (src.length < 4) return src.map((point) => point.clone());
  const control = bsplineControlPolygon(src, opts.controlSpacing ?? 2.25);
  const refined = optimizeBsplineControls(control, opts);
  const sampled = sampleBsplineControls(refined, opts.samplesPerSpan ?? 6).map(clampTrajectoryPoint);
  return enforceTrajectoryClearance(sampled, opts.clearanceMargin ?? 1.18, opts.clearancePasses ?? 2);
}

function optimizeFastPlannerTrajectory(route) {
  const seed = route.smooth?.length ? route.smooth : route.raw;
  return optimizeBspline(seed, {
    controlSpacing: 1.85,
    iterations: 58,
    step: 0.064,
    smoothWeight: 0.86,
    guideWeight: 0.08,
    obstacleMargin: 1.5,
    obstacleWeight: 2.35,
    feasibilityWeight: 0.34,
    maxControlSpacing: 2.55,
    samplesPerSpan: 7,
    clearanceMargin: 1.16,
    clearancePasses: 3,
  });
}

function samplePolynomialTrajectoryWithDurations(points, durations, order) {
  const waypoints = uniqueTrajectoryPoints(points);
  if (waypoints.length < 2) return waypoints;
  const clippedDurations = durations.map((duration, i) => {
    const fallback = waypoints[i].distanceTo(waypoints[i + 1]) / 3.1;
    return clamp(Number.isFinite(duration) ? duration : fallback, 0.35, 3.6);
  });
  const derivatives = waypointDerivatives(waypoints, clippedDurations, order === "snap");
  const out = [];

  for (let i = 0; i < waypoints.length - 1; i += 1) {
    const T = clippedDurations[i];
    const coeffs = { x: null, y: null, z: null };
    for (const axis of ["x", "y", "z"]) {
      coeffs[axis] =
        order === "snap"
          ? septicHermiteCoefficients(
              waypoints[i][axis],
              waypoints[i + 1][axis],
              derivatives.velocity[i][axis],
              derivatives.velocity[i + 1][axis],
              derivatives.accel[i][axis],
              derivatives.accel[i + 1][axis],
              derivatives.jerk[i][axis],
              derivatives.jerk[i + 1][axis],
              T,
            )
          : quinticHermiteCoefficients(
              waypoints[i][axis],
              waypoints[i + 1][axis],
              derivatives.velocity[i][axis],
              derivatives.velocity[i + 1][axis],
              derivatives.accel[i][axis],
              derivatives.accel[i + 1][axis],
              T,
            );
    }
    const samples = Math.max(5, Math.ceil(waypoints[i].distanceTo(waypoints[i + 1]) / 0.74));
    out.push(...samplePolynomialSegment(coeffs, T, samples, i === 0));
  }
  return out;
}

function selectMincoWaypoints(points) {
  const reference = shortcutPoints(uniqueTrajectoryPoints(points), 0.56);
  const dense = reference.length >= 4 ? reference : densifyPolyline(points, 2.4);
  const cum = polylineCumulative(dense);
  const total = cum[cum.length - 1] || 1;
  const count = Math.min(12, Math.max(4, Math.ceil(total / 7.2) + 1));
  const out = [];
  for (let i = 0; i < count; i += 1) {
    out.push(samplePolylineByDistance(dense, cum, (total * i) / (count - 1)));
  }
  return uniqueTrajectoryPoints(out);
}

function optimizeMincoWaypoints(points) {
  const waypoints = selectMincoWaypoints(points).map((point) => point.clone());
  const guide = waypoints.map((point) => point.clone());
  const durations = segmentDurations(waypoints, 3.25);
  const grad = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const prevDir = new THREE.Vector3();
  const nextDir = new THREE.Vector3();

  for (let iter = 0; iter < 46; iter += 1) {
    for (let s = 0; s < durations.length; s += 1) {
      const length = waypoints[s].distanceTo(waypoints[s + 1]);
      const mid = waypoints[s].clone().lerp(waypoints[s + 1], 0.5);
      const clearancePenalty = clamp((1.45 - Math.min(sampleEsdf(mid), sampleEsdf(waypoints[s]), sampleEsdf(waypoints[s + 1]))) / 1.45, 0, 0.55);
      let turnPenalty = 0;
      if (s > 0 && s < waypoints.length - 1) {
        prevDir.copy(waypoints[s]).sub(waypoints[s - 1]);
        nextDir.copy(waypoints[s + 1]).sub(waypoints[s]);
        const denom = Math.max(prevDir.length() * nextDir.length(), 0.001);
        turnPenalty = Math.acos(clamp(prevDir.dot(nextDir) / denom, -1, 1)) * 0.22;
      }
      const target = clamp((length / 3.25) * (1 + turnPenalty + clearancePenalty), 0.38, 3.5);
      durations[s] = THREE.MathUtils.lerp(durations[s], target, 0.2);
    }

    for (let i = 1; i < waypoints.length - 1; i += 1) {
      grad.copy(waypoints[i]).multiplyScalar(2).sub(waypoints[i - 1]).sub(waypoints[i + 1]).multiplyScalar(0.78);
      grad.addScaledVector(waypoints[i].clone().sub(guide[i]), 0.12);

      const prevRate = waypoints[i].distanceTo(waypoints[i - 1]) / Math.max(durations[i - 1], 0.001);
      const nextRate = waypoints[i + 1].distanceTo(waypoints[i]) / Math.max(durations[i], 0.001);
      prevDir.copy(waypoints[i]).sub(waypoints[i - 1]);
      nextDir.copy(waypoints[i]).sub(waypoints[i + 1]);
      if (prevDir.lengthSq() > 0.001) grad.addScaledVector(prevDir.normalize(), (prevRate - 3.25) * 0.05);
      if (nextDir.lengthSq() > 0.001) grad.addScaledVector(nextDir.normalize(), (nextRate - 3.25) * 0.05);

      const clearance = sampleEsdf(waypoints[i]);
      if (clearance < 1.48) {
        esdfGradient(waypoints[i], tmp);
        grad.addScaledVector(tmp, -(1.48 - clearance) * 2.45);
      }
      if (segmentBlocked(waypoints[i - 1], waypoints[i], 0.58) || segmentBlocked(waypoints[i], waypoints[i + 1], 0.58)) {
        esdfGradient(waypoints[i], tmp);
        grad.addScaledVector(tmp, -1.8);
        grad.y -= 0.45;
      }

      const move = grad.multiplyScalar(-0.075);
      if (move.length() > 0.5) move.setLength(0.5);
      waypoints[i].add(move);
      clampTrajectoryPoint(waypoints[i]);
    }
  }

  return { waypoints, durations };
}

function optimizeMincoTrajectory(points) {
  const src = uniqueTrajectoryPoints(points);
  if (src.length < 4) return optimizeMinimumSnapTrajectory(src);
  const { waypoints, durations } = optimizeMincoWaypoints(src);
  const trajectory = samplePolynomialTrajectoryWithDurations(waypoints, durations, "snap");
  return enforceTrajectoryClearance(trajectory.map(clampTrajectoryPoint), 1.18, 3);
}

function curvatureSpeedProfile(points, clearanceAware = false) {
  const n = points.length;
  const profile = new Array(n).fill(1);
  for (let i = 1; i < n - 1; i += 1) {
    const a = points[i].clone().sub(points[i - 1]);
    const b = points[i + 1].clone().sub(points[i]);
    const la = a.length();
    const lb = b.length();
    if (la < 0.001 || lb < 0.001) continue;
    const angle = Math.acos(clamp(a.dot(b) / (la * lb), -1, 1));
    profile[i] = clamp(1.18 - angle * 2.1, 0.55, 1.18);
    if (clearanceAware) profile[i] = Math.min(profile[i], clamp(0.52 + sampleEsdf(points[i]) * 0.26, 0.52, 1.18));
  }
  for (let pass = 0; pass < 3; pass += 1) {
    for (let i = 1; i < n - 1; i += 1) {
      profile[i] = Math.min(profile[i], profile[i - 1] + 0.08, profile[i + 1] + 0.08);
    }
  }
  return profile;
}

// --- route dispatch and cache ---

const routeCache = new Map();
let planStartTime = 0;
let planBudgetExceeded = false;

function planRouteForAlgorithm(id, start) {
  if (id === "A01") {
    const cells = planGridAStar(start);
    return cells && routeFromCells(cells);
  }
  if (id === "A02") {
    const cells = planHybridAStar(start);
    return cells && routeFromCells(cells);
  }
  if (id === "A03") {
    const cells = planJumpPointSearch(start);
    return cells && routeFromCells(cells);
  }
  if (id === "A04" || id === "A05") {
    const points = planRrtStar(start, id === "A05");
    return points && routeFromPoints(points);
  }
  if (id === "A06") {
    const points = planBitStar(start);
    return points && routeFromPoints(points);
  }
  if (id === "A07") {
    const cells = planKinodynamicAStar(start, false);
    return cells && routeFromCells(cells);
  }
  if (id === "B04" || id === "B06") {
    const cells = planGridAStar(start);
    return cells && routeFromCells(cells);
  }
  if (id === "B05") {
    const cells = planKinodynamicAStar(start, false) ?? planGridAStar(start);
    return cells && routeFromCells(cells);
  }
  if (id === "A08") {
    const cells = planMotionPrimitiveSearch(start);
    return cells && routeFromCells(cells);
  }
  return null;
}

function postProcessRoute(id, route) {
  if (id === "B01") {
    route.smooth = optimizeMinimumSnapTrajectory(sparseTrajectoryWaypoints(route));
  } else if (id === "B02") {
    route.smooth = optimizeMinimumJerkTrajectory(sparseTrajectoryWaypoints(route));
  } else if (id === "B03") {
    route.smooth = optimizeSafeCorridorTrajectory(densifyPolyline(sparseTrajectoryWaypoints(route), 1.0), 1.55);
  } else if (id === "B04") {
    route.smooth = optimizeBspline(route.smooth);
  } else if (id === "B05") {
    route.smooth = optimizeFastPlannerTrajectory(route);
  } else if (id === "B06") {
    route.smooth = optimizeMincoTrajectory(route.smooth);
  }
  if (id === "B05" || id === "B06") {
    route.speedScale = curvatureSpeedProfile(route.smooth, true);
  }
  return route;
}

function computeRoute(start) {
  const id = state.algorithmId;
  const sampling = id === "A04" || id === "A05" || id === "A06";
  const key = sampling ? `${id}:${start.x},${start.y},${start.z & ~1}` : `${id}:${start.index}`;
  const cached = routeCache.get(key);
  if (cached) return cached;
  if (!planBudgetExceeded && performance.now() - planStartTime > 1600) planBudgetExceeded = true;
  let route = null;
  if (!planBudgetExceeded) {
    try {
      route = planRouteForAlgorithm(id, start);
    } catch (error) {
      route = null;
    }
  }
  if (!route) route = tracePathFromCell(start);
  route = postProcessRoute(id, route);
  routeCache.set(key, route);
  return route;
}

function startCellForDrone(index, rng) {
  const grid = state.grid;
  const profile = getProfile();
  const layerCount = profile.laneCount ? clamp(profile.laneCount, 4, 8) : state.mode === "central" ? 6 : 5;
  const layers = Array.from({ length: layerCount }, (_, layer) => clamp(layer + 1, 1, grid.ny - 1));
  const band = index % Math.max(1, Math.floor(Math.sqrt(state.count)));
  const zNorm = state.count === 1 ? 0.5 : (band + rng() * 0.65) / Math.max(1, Math.sqrt(state.count));
  const z = clamp(Math.floor(4 + zNorm * (grid.nz - 8)), 3, grid.nz - 4);
  const y = layers[index % layers.length];
  const x = 2 + Math.floor(rng() * 3);
  return nearestFreeCell({ x, y, z }, true);
}

function spawnDrones() {
  const algorithm = getAlgorithm();
  const profile = getProfile();
  const rng = createRng(scenarioSeed(state.scenario) + scenarioSeed(algorithm.id) + state.count * 31 + 1009);
  state.elapsed = 0;
  state.drones = [];
  routeCache.clear();
  planStartTime = performance.now();
  planBudgetExceeded = false;

  for (let i = 0; i < state.count; i += 1) {
    const start = startCellForDrone(i, rng);
    const route = computeRoute(start);
    const laneCount = profile.laneCount ?? 5;
    const laneLift = state.mode === "central" ? ((i % laneCount) - (laneCount - 1) / 2) * 0.22 : 0;
    const points = route.smooth.map((point) => point.clone().add(new THREE.Vector3(0, laneLift, 0)));
    if (points.length < 2) {
      points.push(points[0].clone().add(new THREE.Vector3(8, 0, 0)));
    }
    const color = agentBaseColor.clone().lerp(agentHighlightColor, (i % 7) * 0.035);
    const offset = new THREE.Vector3((rng() - 0.5) * 0.45, (rng() - 0.5) * 0.28, (rng() - 0.5) * 0.45);
    const position = points[0].clone().add(offset);
    state.drones.push({
      id: i,
      position,
      velocity: new THREE.Vector3(),
      desired: new THREE.Vector3(),
      avoidance: new THREE.Vector3(),
      policy: new THREE.Vector3(),
      path: points,
      rawPath: route.raw.map((point) => point.clone()),
      cells: route.cells ?? null,
      speedScale: route.speedScale ? route.speedScale.slice() : null,
      waypoint: 1,
      speed: (2.2 + rng() * 1.1) * (profile.speed ?? 1),
      radius: 0.36,
      safety: (1.12 + (i % 4) * 0.04) * (profile.safetyScale ?? 1),
      color,
      startDelay:
        state.mode === "central"
          ? (i % 22) * (profile.startDelay ?? 0.07)
          : rng() * 0.35 + (i % 11) * (profile.startDelay ?? 0),
      trail: [position.clone()],
      complete: false,
      neighborCount: 0,
    });
  }

  state.selected = Math.min(state.selected, state.count - 1);
  coordinateCentral();
  rebuildInstanceMeshes();
  rebuildPathLines();
  rebuildSelectedVisuals();
}

function rebuildInstanceMeshes() {
  scene.remove(droneMesh);
  droneMesh.dispose();
  droneMesh = new THREE.InstancedMesh(droneGeometry, droneMaterial, state.count);
  droneMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  droneMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(state.count * 3), 3);
  droneMesh.frustumCulled = false;
  scene.add(droneMesh);

  scene.remove(glowMesh);
  glowMesh.dispose();
  glowMesh = new THREE.InstancedMesh(glowGeometry, glowMaterial, state.count);
  glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  glowMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(state.count * 3), 3);
  glowMesh.frustumCulled = false;
  scene.add(glowMesh);

  scene.remove(safetyMesh);
  safetyMesh.dispose();
  safetyMesh = new THREE.InstancedMesh(safetyGeometry, safetyMaterial, state.count);
  safetyMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  safetyMesh.frustumCulled = false;
  safetyMesh.visible = state.showSafety;
  scene.add(safetyMesh);

  state.drones.forEach((drone, i) => {
    droneMesh.setColorAt(i, drone.color);
    glowMesh.setColorAt(i, drone.color.clone().lerp(agentGlowColor, 0.62));
  });
  droneMesh.instanceColor.needsUpdate = true;
  glowMesh.instanceColor.needsUpdate = true;
}

function rebuildPathLines() {
  if (pathLines) {
    pathGroup.remove(pathLines);
    pathLines.geometry.dispose();
    pathLines.material.dispose();
    pathLines = null;
  }

  const positions = [];
  const colors = [];
  for (const drone of state.drones) {
    const path = drone.path;
    const lineColor = drone.color.clone().lerp(modeTint[state.mode], 0.36);
    for (let i = 0; i < path.length - 1; i += 1) {
      positions.push(path[i].x, path[i].y, path[i].z, path[i + 1].x, path[i + 1].y, path[i + 1].z);
      colors.push(lineColor.r, lineColor.g, lineColor.b, lineColor.r, lineColor.g, lineColor.b);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: state.count > 500 ? 0.22 : state.count > 100 ? 0.34 : 0.58,
    depthWrite: false,
  });
  pathLines = new THREE.LineSegments(geometry, material);
  pathLines.visible = state.showPaths;
  pathGroup.add(pathLines);
}

function rebuildSelectedVisuals() {
  if (selectedPathLine) {
    selectedGroup.remove(selectedPathLine);
    selectedPathLine.geometry.dispose();
    selectedPathLine.material.dispose();
  }
  if (selectedRawLine) {
    selectedGroup.remove(selectedRawLine);
    selectedRawLine.geometry.dispose();
    selectedRawLine.material.dispose();
  }
  clearGroup(corridorGroup);

  const drone = state.drones[state.selected];
  if (!drone) return;

  selectedPathLine = makeLineFromPoints(drone.path, 0xffffff, 0.95);
  selectedGroup.add(selectedPathLine);

  selectedRawLine = makeLineFromPoints(drone.rawPath, 0xffbf47, state.mode === "search" ? 0.72 : 0.24);
  selectedRawLine.visible = state.mode === "search";
  selectedGroup.add(selectedRawLine);

  rebuildCorridor(drone);
  updateSelectedReadout();
}

function makeLineFromPoints(points, color, opacity) {
  const positions = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    positions.push(points[i].x, points[i].y, points[i].z, points[i + 1].x, points[i + 1].y, points[i + 1].z);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  return new THREE.LineSegments(geometry, material);
}

function rebuildCorridor(drone) {
  clearGroup(corridorGroup);
  if (!state.showCorridor) return;
  const material = new THREE.MeshBasicMaterial({
    color: state.mode === "optimize" ? 0x76e06f : 0x2ee6d6,
    transparent: true,
    opacity: state.mode === "optimize" ? 0.12 : 0.075,
    wireframe: true,
  });

  const section = 2.4 * (getProfile().corridor ?? 1);
  const path = drone.path;
  const stride = Math.max(1, Math.floor(path.length / 12));
  for (let i = 0; i < path.length - 1; i += stride) {
    const a = path[i];
    const b = path[Math.min(i + stride, path.length - 1)];
    const direction = b.clone().sub(a);
    const length = direction.length();
    if (length < 0.05) continue;
    const geometry = new THREE.BoxGeometry(length, section, section);
    const mesh = new THREE.Mesh(geometry, material.clone());
    mesh.position.copy(a).add(b).multiplyScalar(0.5);
    tempQuaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), direction.normalize());
    mesh.quaternion.copy(tempQuaternion);
    corridorGroup.add(mesh);
  }
  corridorGroup.visible = state.showCorridor && (state.mode === "optimize" || state.mode === "avoid" || state.count <= 50);
}

function rebuildGridOverlay() {
  if (gridLines) {
    overlayGroup.remove(gridLines);
    gridLines.geometry.dispose();
    gridLines.material.dispose();
  }

  const grid = state.grid;
  const positions = [];
  const minX = -grid.nx * grid.cell * 0.5;
  const maxX = grid.nx * grid.cell * 0.5;
  const minZ = -grid.nz * grid.cell * 0.5;
  const maxZ = grid.nz * grid.cell * 0.5;
  for (let y = 0; y < grid.ny; y += 1) {
    const yy = grid.yBase + y * grid.yStep;
    for (let x = 0; x <= grid.nx; x += 2) {
      const xx = minX + x * grid.cell;
      positions.push(xx, yy, minZ, xx, yy, maxZ);
    }
    for (let z = 0; z <= grid.nz; z += 2) {
      const zz = minZ + z * grid.cell;
      positions.push(minX, yy, zz, maxX, yy, zz);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0x42505a,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  });
  gridLines = new THREE.LineSegments(geometry, material);
  gridLines.visible = state.showGrid;
  overlayGroup.add(gridLines);
}

function rebuildSearchCloud() {
  if (searchCloud) {
    overlayGroup.remove(searchCloud);
    searchCloud.geometry.dispose();
    searchCloud.material.dispose();
  }

  const grid = state.grid;
  const positions = [];
  const colors = [];
  let maxDistance = 1;
  for (let i = 0; i < state.distance.length; i += 1) {
    if (Number.isFinite(state.distance[i])) maxDistance = Math.max(maxDistance, state.distance[i]);
  }
  for (let i = 0; i < grid.total; i += 5) {
    if (!grid.passable[i] || !Number.isFinite(state.distance[i])) continue;
    const c = grid.unpack(i);
    if ((c.x + c.y + c.z) % 3 !== 0) continue;
    const p = grid.cellToWorld(c.x, c.y, c.z);
    positions.push(p.x, p.y, p.z);
    const t = state.distance[i] / maxDistance;
    const color = new THREE.Color().setHSL(0.48 - t * 0.34, 0.75, 0.58);
    colors.push(color.r, color.g, color.b);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 0.07,
    vertexColors: true,
    transparent: true,
    opacity: 0.56,
    depthWrite: false,
  });
  searchCloud = new THREE.Points(geometry, material);
  searchCloud.visible = state.mode === "search" || state.showGrid;
  overlayGroup.add(searchCloud);
}

function updateModeVisuals() {
  const algorithm = getAlgorithm();
  if (pathLines) {
    pathLines.material.opacity = state.count > 500 ? 0.22 : state.count > 100 ? 0.34 : 0.58;
  }
  if (searchCloud) {
    searchCloud.visible = state.mode === "search" || state.showGrid;
    searchCloud.material.opacity = state.mode === "search" ? 0.56 : 0.22;
  }
  if (selectedRawLine) {
    selectedRawLine.visible = state.mode === "search";
  }
  corridorGroup.visible = state.showCorridor && (state.mode === "optimize" || state.mode === "avoid" || state.count <= 50);
  droneMaterial.emissive.copy(agentEmissiveColor);
  droneMaterial.emissiveIntensity = 0.62;
  glowMaterial.opacity = state.count > 500 ? 0.2 : 0.26;
  selectedMaterial.color.copy(modeTint[algorithm.mode]).lerp(new THREE.Color("#ffffff"), 0.36);
  rebuildPathLines();
  rebuildSelectedVisuals();
}

function resetScene() {
  state.elapsed = 0;
  buildScenario(state.scenario);
  rebuildGrid();
  spawnDrones();
  updateModeVisuals();
}

function hashPosition(position, cellSize) {
  const x = Math.floor(position.x / cellSize);
  const y = Math.floor(position.y / cellSize);
  const z = Math.floor(position.z / cellSize);
  return `${x},${y},${z}`;
}

function buildSpatialHash(cellSize) {
  const buckets = new Map();
  for (const drone of state.drones) {
    const key = hashPosition(drone.position, cellSize);
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = [];
      buckets.set(key, bucket);
    }
    bucket.push(drone);
  }
  return buckets;
}

function queryNeighbors(buckets, drone, cellSize) {
  const baseX = Math.floor(drone.position.x / cellSize);
  const baseY = Math.floor(drone.position.y / cellSize);
  const baseZ = Math.floor(drone.position.z / cellSize);
  const neighbors = [];
  for (let x = baseX - 1; x <= baseX + 1; x += 1) {
    for (let y = baseY - 1; y <= baseY + 1; y += 1) {
      for (let z = baseZ - 1; z <= baseZ + 1; z += 1) {
        const bucket = buckets.get(`${x},${y},${z}`);
        if (bucket) neighbors.push(...bucket);
      }
    }
  }
  return neighbors;
}

function updateSimulation(dt) {
  if (!state.running) return;
  state.elapsed += dt;
  const profile = getProfile();
  const baseCell = profile.avoidRange ? profile.avoidRange * 1.65 : state.mode === "field" ? 4.8 : 3.4;
  const hashCell = Math.max(baseCell, profile.perceptionRadius ?? 0);
  const buckets = buildSpatialHash(hashCell);
  let conflicts = 0;
  let spacingSum = 0;
  let spacingSamples = 0;

  for (const drone of state.drones) {
    if (state.elapsed < drone.startDelay) continue;
    advanceWaypoint(drone);
    computeDesiredVelocity(drone);
    drone.avoidance.set(0, 0, 0);
    drone.neighborCount = 0;

    const neighbors = queryNeighbors(buckets, drone, hashCell);
    const nearest = computeSwarmAvoidance(drone, neighbors);
    if (nearest < Number.POSITIVE_INFINITY) {
      spacingSum += nearest;
      spacingSamples += 1;
      if (nearest < drone.radius * 2.45) conflicts += 1;
    }

    computeObstacleAvoidance(drone);
    applyModeForces(drone, neighbors, dt);
    integrateDrone(drone, dt);
  }

  state.conflicts = conflicts;
  state.averageSpacing = spacingSamples ? spacingSum / spacingSamples : 0;
}

function advanceWaypoint(drone) {
  if (drone.path.length < 2) return;
  const windowEnd = Math.min(drone.path.length - 1, drone.waypoint + 6);
  let nearestIndex = drone.waypoint;
  let nearestDistance = drone.position.distanceTo(drone.path[drone.waypoint]);
  for (let i = drone.waypoint + 1; i <= windowEnd; i += 1) {
    const distance = drone.position.distanceTo(drone.path[i]);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = i;
    }
  }
  drone.waypoint = nearestIndex;
  const next = drone.path[Math.min(drone.waypoint + 1, drone.path.length - 1)];
  const capture = Math.max(0.72, drone.path[drone.waypoint].distanceTo(next) * 0.45);
  if (nearestDistance < capture) {
    drone.waypoint += 1;
    if (drone.waypoint >= drone.path.length) {
      drone.path.reverse();
      drone.rawPath.reverse();
      if (drone.speedScale) drone.speedScale.reverse();
      drone.waypoint = 1;
      drone.startDelay = state.elapsed + 0.15 + (drone.id % 17) * 0.025;
    }
  }
}

function computeDesiredVelocity(drone) {
  const profile = getProfile();
  const lookahead = profile.lookahead ?? (state.mode === "optimize" ? 2 : 1);
  const lookaheadIndex = Math.min(drone.path.length - 1, drone.waypoint + lookahead);
  const target = drone.path[lookaheadIndex];
  const desired = target.clone().sub(drone.position);
  if (desired.lengthSq() < 0.0001) {
    drone.desired.set(0, 0, 0);
    return;
  }
  desired.normalize();
  let speed = drone.speed;
  if (drone.speedScale) speed *= drone.speedScale[Math.min(drone.waypoint, drone.speedScale.length - 1)];
  if (state.mode === "central") {
    const phase = (drone.id % 9) / 9;
    speed *= 0.86 + phase * 0.22;
  }
  if (state.mode === "field") speed *= 0.9;
  if (profile.wander) {
    const wave = Math.sin(state.elapsed * 1.7 + drone.id * 0.61) * profile.wander;
    desired.x += Math.sin(drone.id * 1.17) * wave;
    desired.z += Math.cos(drone.id * 1.37) * wave;
    desired.normalize();
  }
  drone.desired.copy(desired.multiplyScalar(speed));
  drone.policy.copy(drone.desired);
}

function isAlgorithm(...ids) {
  return ids.includes(state.algorithmId);
}

function computeBoidsForces(drone, neighbors) {
  const profile = getProfile();
  const perceptionRadius = profile.perceptionRadius ?? 6;
  const separationRadius = profile.avoidRange ?? 3;
  const center = new THREE.Vector3();
  const averageVelocity = new THREE.Vector3();
  const separation = new THREE.Vector3();
  let count = 0;
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const offset = drone.position.clone().sub(other.position);
    const distance = offset.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > perceptionRadius) continue;

    center.add(other.position);
    averageVelocity.add(other.velocity);
    count += 1;

    if (distance < separationRadius) {
      const strength = (separationRadius - distance) / separationRadius;
      separation.add(offset.normalize().multiplyScalar(strength * strength));
      drone.neighborCount += 1;
    }
  }

  if (!count) return nearest;

  center.multiplyScalar(1 / count);
  averageVelocity.multiplyScalar(1 / count);

  const cohesion = center.sub(drone.position);
  if (cohesion.lengthSq() > 0.0001) {
    cohesion.setLength(drone.speed).sub(drone.velocity).multiplyScalar(profile.cohesionWeight ?? 0.3);
    drone.avoidance.add(cohesion);
  }

  if (averageVelocity.lengthSq() > 0.0001) {
    averageVelocity.setLength(drone.speed).sub(drone.velocity).multiplyScalar(profile.alignmentWeight ?? 0.5);
    drone.avoidance.add(averageVelocity);
  }

  if (separation.lengthSq() > 0.0001) {
    separation.multiplyScalar(profile.separationWeight ?? 4);
    drone.avoidance.add(separation);
  }

  return nearest;
}

function computeApfForces(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 3.2;
  const gain = profile.avoidWeight ?? 3.1;
  let nearest = Number.POSITIVE_INFINITY;
  for (const other of neighbors) {
    if (other === drone) continue;
    const offset = drone.position.clone().sub(other.position);
    const distance = offset.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > range) continue;
    const strength = (range - distance) / range;
    drone.avoidance.add(offset.normalize().multiplyScalar(strength * strength * gain));
    drone.neighborCount += 1;
  }
  return nearest;
}
function computeSocialForceInteractions(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 4.5;
  const amplitude = profile.socialA ?? 4.2;
  const decay = profile.socialB ?? 0.9;
  let nearest = Number.POSITIVE_INFINITY;

  drone.avoidance.add(drone.desired.clone().sub(drone.velocity).multiplyScalar(1 / (profile.socialTau ?? 0.75)));

  for (const other of neighbors) {
    if (other === drone) continue;
    const offset = drone.position.clone().sub(other.position);
    const distance = offset.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > range) continue;

    const desiredSpacing = drone.safety * 0.72 + other.safety * 0.72;
    const force = amplitude * Math.exp((desiredSpacing - distance) / decay);
    const tangentialDamping = other.velocity.clone().sub(drone.velocity).multiplyScalar(0.08);
    drone.avoidance.add(offset.normalize().multiplyScalar(force)).add(tangentialDamping);
    drone.neighborCount += 1;
  }

  return nearest;
}

function perpendicularUnit(axis, seed = 0) {
  const base = Math.abs(axis.y) < 0.82 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const tangent = base.addScaledVector(axis, -base.dot(axis));
  if (tangent.lengthSq() < 0.0001) tangent.set(-axis.z, 0, axis.x);
  tangent.normalize();
  if (seed % 2) tangent.multiplyScalar(-1);
  return tangent;
}

function rvoCollisionTime(velocity, constraint) {
  if (constraint.distance <= constraint.radius) return 0;
  const relVelocity = velocity.clone().sub(constraint.apex);
  const relSpeedSq = relVelocity.lengthSq();
  if (relSpeedSq < 0.0001) return Number.POSITIVE_INFINITY;
  const closestTime = clamp(constraint.relPos.dot(relVelocity) / relSpeedSq, 0, constraint.horizon);
  const closest = constraint.relPos.clone().sub(relVelocity.multiplyScalar(closestTime));
  return closest.lengthSq() < constraint.radiusSq ? closestTime : Number.POSITIVE_INFINITY;
}

function clampVelocity(velocity, maxSpeed) {
  const clamped = velocity.clone();
  if (clamped.length() > maxSpeed) clamped.setLength(maxSpeed);
  return clamped;
}

function projectVelocityOutOfRvo(velocity, constraint, seed) {
  const relVelocity = velocity.clone().sub(constraint.apex);
  const relSpeed = Math.max(relVelocity.length(), 0.12);
  const parallel = relVelocity.dot(constraint.axis);
  const perpendicular = relVelocity.clone().addScaledVector(constraint.axis, -parallel);
  const perpendicularDir =
    perpendicular.lengthSq() > 0.0001 ? perpendicular.normalize() : perpendicularUnit(constraint.axis, seed);
  const coneAngle = Math.asin(clamp(constraint.radius / Math.max(constraint.distance, constraint.radius + 0.001), 0.02, 0.92));
  const boundaryAngle = clamp(coneAngle + 0.075, 0.04, 1.42);
  const boundaryDir = constraint.axis
    .clone()
    .multiplyScalar(Math.cos(boundaryAngle))
    .addScaledVector(perpendicularDir, Math.sin(boundaryAngle))
    .normalize();
  const escapeSpeed = Math.max(relSpeed, Math.max(0.15, (constraint.distance - constraint.radius) / constraint.horizon));
  return constraint.apex.clone().addScaledVector(boundaryDir, escapeSpeed);
}

function addRvoDirectionSamples(candidates, desired, maxSpeed, seed) {
  const forward = desired.lengthSq() > 0.0001 ? desired.clone().normalize() : new THREE.Vector3(1, 0, 0);
  const side = perpendicularUnit(forward, seed);
  const lift = new THREE.Vector3().crossVectors(forward, side);
  if (lift.lengthSq() < 0.0001) lift.set(0, 1, 0);
  lift.normalize();
  const dirs = [
    forward.clone(),
    forward.clone().addScaledVector(side, 0.52).normalize(),
    forward.clone().addScaledVector(side, -0.52).normalize(),
    forward.clone().addScaledVector(lift, 0.48).normalize(),
    forward.clone().addScaledVector(lift, -0.48).normalize(),
    side.clone(),
    side.clone().multiplyScalar(-1),
    lift.clone(),
    lift.clone().multiplyScalar(-1),
  ];
  for (const scale of [0.36, 0.68, 1]) {
    for (const dir of dirs) candidates.push(dir.clone().multiplyScalar(maxSpeed * scale));
  }
}

function scoreRvoVelocity(velocity, desired, constraints, maxSpeed) {
  let score = velocity.distanceToSquared(desired);
  const speedOver = Math.max(0, velocity.length() - maxSpeed);
  if (speedOver > 0) score += speedOver * speedOver * 80;

  let safe = true;
  for (const constraint of constraints) {
    const collisionTime = rvoCollisionTime(velocity, constraint);
    if (collisionTime !== Number.POSITIVE_INFINITY) {
      safe = false;
      const relVelocity = velocity.clone().sub(constraint.apex);
      const closest = constraint.relPos.clone().sub(relVelocity.multiplyScalar(collisionTime));
      const clearance = Math.sqrt(Math.max(closest.lengthSq(), 0.0001));
      const depth = Math.max(0, constraint.radius - clearance);
      score += 900 + depth * depth * 180 + (constraint.horizon - collisionTime) * 18;
    }
  }

  return { safe, score };
}

function projectOrcaVelocity(preferred, constraints, maxSpeed) {
  const velocity = clampVelocity(preferred, maxSpeed);
  const corrections = constraints.map(() => new THREE.Vector3());

  for (let iter = 0; iter < 12; iter += 1) {
    for (let i = 0; i < constraints.length; i += 1) {
      const constraint = constraints[i];
      const y = velocity.clone().add(corrections[i]);
      const violation = constraint.point.dot(constraint.normal) - y.dot(constraint.normal);
      const projected = violation > 0 ? y.clone().addScaledVector(constraint.normal, violation) : y.clone();
      corrections[i].copy(y).sub(projected);
      velocity.copy(projected);
    }
    if (velocity.length() > maxSpeed) velocity.setLength(maxSpeed);
  }

  return velocity;
}

function makeOrcaConstraint(drone, other, horizon, radius) {
  const relPos = other.position.clone().sub(drone.position);
  const distance = relPos.length();
  if (distance < 0.001) return null;
  const current = {
    relPos,
    distance,
    axis: relPos.clone().multiplyScalar(1 / distance),
    radius,
    radiusSq: radius * radius,
    horizon,
    apex: other.velocity.clone(),
  };

  let normal;
  let u;
  if (distance <= radius * 1.02) {
    normal = drone.position.clone().sub(other.position).normalize();
    u = normal.clone().multiplyScalar((radius - distance + 0.05) / 0.18);
  } else if (rvoCollisionTime(drone.velocity, current) === Number.POSITIVE_INFINITY) {
    return null;
  } else {
    const escape = projectVelocityOutOfRvo(drone.velocity, current, drone.id + other.id);
    u = escape.sub(drone.velocity);
    if (u.lengthSq() < 0.0001) return null;
    normal = u.clone().normalize();
  }

  return {
    normal,
    point: drone.velocity.clone().addScaledVector(u, 0.5),
  };
}

function computeOrca3dAvoidance(drone, neighbors) {
  const profile = getProfile();
  const horizon = profile.timeHorizon ?? 2.6;
  const maxSpeed = drone.speed * 1.14;
  const maxNeighborDistance = Math.max(profile.avoidRange ?? 2.65, drone.speed * horizon + 1.5);
  const constraints = [];
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const distance = drone.position.distanceTo(other.position);
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > maxNeighborDistance) continue;
    if (distance < (profile.avoidRange ?? 2.65)) drone.neighborCount += 1;
    const radius = Math.max(drone.radius + other.radius + 0.18, (drone.safety + other.safety) * 0.48);
    const constraint = makeOrcaConstraint(drone, other, horizon, radius);
    if (constraint) constraints.push(constraint);
  }

  if (!constraints.length) return nearest;
  let best = projectOrcaVelocity(drone.desired, constraints, maxSpeed);

  const axes = makeDmpcAxes(drone.desired, drone.velocity, drone.id);
  let bestCost = best.distanceToSquared(drone.desired);
  for (const axis of axes) {
    for (const sign of [-1, 1]) {
      const candidate = projectOrcaVelocity(drone.desired.clone().addScaledVector(axis, sign * maxSpeed * 0.28), constraints, maxSpeed);
      const cost = candidate.distanceToSquared(drone.desired);
      if (cost < bestCost) {
        best = candidate;
        bestCost = cost;
      }
    }
  }

  drone.avoidance.add(best.sub(drone.desired));
  return nearest;
}

function hrvoApexFor(drone, other, axis, radius, horizon) {
  const voApex = other.velocity.clone();
  const rvoApex = drone.velocity.clone().add(other.velocity).multiplyScalar(0.5);
  const preferredRel = drone.desired.clone().sub(rvoApex);
  const tangent = preferredRel.clone().addScaledVector(axis, -preferredRel.dot(axis));
  if (tangent.lengthSq() < 0.0001) tangent.copy(perpendicularUnit(axis, drone.id + other.id));
  tangent.normalize();
  const side = tangent.dot(preferredRel) >= 0 ? 1 : -1;
  return voApex
    .lerp(rvoApex, 0.68)
    .addScaledVector(tangent, side * (radius / Math.max(horizon, 0.1)) * 0.42);
}

function computeHrvo3dAvoidance(drone, neighbors) {
  const profile = getProfile();
  const horizon = profile.timeHorizon ?? 2.6;
  const maxSpeed = drone.speed * 1.16;
  const maxNeighborDistance = Math.max(profile.avoidRange ?? 2.65, drone.speed * horizon + 1.5);
  const constraints = [];
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const relPos = other.position.clone().sub(drone.position);
    const distance = relPos.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > maxNeighborDistance) continue;
    if (distance < (profile.avoidRange ?? 2.65)) drone.neighborCount += 1;
    const axis = relPos.clone().multiplyScalar(1 / distance);
    const radius = Math.max(drone.radius + other.radius + 0.15, (drone.safety + other.safety) * 0.47);
    constraints.push({
      relPos,
      distance,
      axis,
      radius,
      radiusSq: radius * radius,
      horizon,
      apex: hrvoApexFor(drone, other, axis, radius, horizon),
    });
  }

  if (!constraints.length) return nearest;
  const candidates = [
    drone.desired.clone(),
    drone.velocity.clone(),
    drone.desired.clone().multiplyScalar(0.7),
    new THREE.Vector3(0, 0, 0),
  ];
  addRvoDirectionSamples(candidates, drone.desired, maxSpeed, drone.id + 29);

  const projected = drone.desired.clone();
  for (let iter = 0; iter < Math.min(20, constraints.length * 3 + 8); iter += 1) {
    let changed = false;
    for (const constraint of constraints) {
      if (rvoCollisionTime(projected, constraint) === Number.POSITIVE_INFINITY) continue;
      projected.copy(projectVelocityOutOfRvo(projected, constraint, drone.id + iter + 41));
      if (projected.length() > maxSpeed) projected.setLength(maxSpeed);
      changed = true;
    }
    if (!changed) break;
  }
  candidates.push(projected.clone());

  for (let i = 0; i < constraints.length; i += 1) {
    candidates.push(projectVelocityOutOfRvo(drone.desired, constraints[i], drone.id + i + 73));
  }

  let best = clampVelocity(candidates[0], maxSpeed);
  let bestEval = scoreRvoVelocity(best, drone.desired, constraints, maxSpeed);
  for (const candidate of candidates) {
    const velocity = clampVelocity(candidate, maxSpeed);
    const evaluation = scoreRvoVelocity(velocity, drone.desired, constraints, maxSpeed);
    if ((evaluation.safe && !bestEval.safe) || evaluation.score < bestEval.score) {
      best = velocity;
      bestEval = evaluation;
    }
  }

  drone.avoidance.add(best.sub(drone.desired));
  return nearest;
}
function computeRvo3dAvoidance(drone, neighbors) {
  const profile = getProfile();
  const horizon = profile.timeHorizon ?? 2.4;
  const maxSpeed = drone.speed * 1.18;
  const maxNeighborDistance = Math.max(profile.avoidRange ?? 2.6, drone.speed * horizon + 1.4);
  const constraints = [];
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const relPos = other.position.clone().sub(drone.position);
    const distance = relPos.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > maxNeighborDistance) continue;
    if (distance < (profile.avoidRange ?? 2.6)) drone.neighborCount += 1;
    const radius = Math.max(drone.radius + other.radius, (drone.safety + other.safety) * 0.48);
    const axis = relPos.clone().multiplyScalar(1 / distance);
    constraints.push({
      relPos,
      distance,
      axis,
      radius,
      radiusSq: radius * radius,
      horizon,
      apex: drone.velocity.clone().add(other.velocity).multiplyScalar(0.5),
    });
  }

  if (!constraints.length) return nearest;

  const candidates = [
    drone.desired.clone(),
    drone.velocity.clone(),
    drone.desired.clone().multiplyScalar(0.72),
    drone.desired.clone().multiplyScalar(0.42),
    new THREE.Vector3(0, 0, 0),
  ];
  addRvoDirectionSamples(candidates, drone.desired, maxSpeed, drone.id);

  const projected = drone.desired.clone();
  for (let iter = 0; iter < Math.min(18, constraints.length * 3 + 6); iter += 1) {
    let changed = false;
    for (const constraint of constraints) {
      if (rvoCollisionTime(projected, constraint) === Number.POSITIVE_INFINITY) continue;
      projected.copy(projectVelocityOutOfRvo(projected, constraint, drone.id + iter));
      if (projected.length() > maxSpeed) projected.setLength(maxSpeed);
      changed = true;
    }
    if (!changed) break;
  }
  candidates.push(projected.clone());

  for (let i = 0; i < constraints.length; i += 1) {
    const constraint = constraints[i];
    candidates.push(projectVelocityOutOfRvo(drone.desired, constraint, drone.id + i));
    candidates.push(projectVelocityOutOfRvo(drone.velocity, constraint, drone.id + i + 17));
  }

  let best = clampVelocity(candidates[0], maxSpeed);
  let bestEval = scoreRvoVelocity(best, drone.desired, constraints, maxSpeed);
  for (const candidate of candidates) {
    const velocity = clampVelocity(candidate, maxSpeed);
    const evaluation = scoreRvoVelocity(velocity, drone.desired, constraints, maxSpeed);
    if ((evaluation.safe && !bestEval.safe) || evaluation.score < bestEval.score) {
      best = velocity;
      bestEval = evaluation;
    }
  }

  drone.avoidance.add(best.sub(drone.desired));
  return nearest;
}

function computeVelocityObstacleAvoidance(drone, neighbors, variant) {
  if (variant === "orca") return computeOrca3dAvoidance(drone, neighbors);
  if (variant === "rvo") return computeRvo3dAvoidance(drone, neighbors);
  if (variant === "hrvo") return computeHrvo3dAvoidance(drone, neighbors);

  const profile = getProfile();
  const timeHorizon = profile.timeHorizon ?? 2.6;
  const reciprocalShare = profile.reciprocalShare ?? 0.5;
  const maxNeighborDistance = Math.max(profile.avoidRange ?? 2.6, drone.speed * timeHorizon + 1.2);
  const candidate = drone.desired.clone();
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const relPos = other.position.clone().sub(drone.position);
    const distance = relPos.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > maxNeighborDistance) continue;

    const apex =
      variant === "hrvo"
        ? other.velocity.clone().lerp(drone.velocity.clone().add(other.velocity).multiplyScalar(0.5), 0.62)
        : other.velocity;
    const relVelocity = candidate.clone().sub(apex);
    const relSpeedSq = Math.max(relVelocity.lengthSq(), 0.0001);
    const closestTime = clamp(relPos.dot(relVelocity) / relSpeedSq, 0, timeHorizon);
    const closest = relPos.clone().sub(relVelocity.clone().multiplyScalar(closestTime));
    const closestDistance = closest.length();
    const combinedRadius = drone.radius + other.radius + 0.68;
    const closingSpeed = Math.max(0, relVelocity.dot(relPos.clone().normalize()));

    if (closestDistance >= combinedRadius && distance >= combinedRadius * 1.18) continue;

    const away =
      closestDistance > 0.001
        ? closest.multiplyScalar(-1 / closestDistance)
        : drone.position.clone().sub(other.position).normalize();
    const penetration = Math.max(0, combinedRadius - closestDistance);
    const urgency = penetration / Math.max(combinedRadius, 0.001);
    const timeScale = Math.max(closestTime, 0.18);
    const correctionMagnitude = penetration / timeScale + closingSpeed * 0.18 + urgency * 0.45;
    candidate.add(away.multiplyScalar(correctionMagnitude * reciprocalShare));

    if (variant === "hrvo" && profile.tangentBias) {
      const tangent = new THREE.Vector3(-away.z, 0, away.x);
      if (tangent.lengthSq() > 0.0001) {
        const side = drone.id % 2 === 0 ? 1 : -1;
        candidate.add(tangent.normalize().multiplyScalar(side * profile.tangentBias * (0.35 + urgency)));
      }
    }
    drone.neighborCount += 1;
  }

  if (candidate.length() > drone.speed * 1.18) candidate.setLength(drone.speed * 1.18);
  drone.avoidance.add(candidate.sub(drone.desired));
  return nearest;
}

function rotatedAroundY(vector, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new THREE.Vector3(vector.x * cos + vector.z * sin, vector.y, -vector.x * sin + vector.z * cos);
}

function projectBufferedVoronoiVelocity(preferred, constraints, maxSpeed) {
  const velocity = clampVelocity(preferred, maxSpeed);
  const corrections = constraints.map(() => new THREE.Vector3());

  for (let iter = 0; iter < 10; iter += 1) {
    for (let i = 0; i < constraints.length; i += 1) {
      const constraint = constraints[i];
      const y = velocity.clone().add(corrections[i]);
      const excess = y.dot(constraint.normal) - constraint.limit;
      const projected = excess > 0 ? y.clone().addScaledVector(constraint.normal, -excess) : y.clone();
      corrections[i].copy(y).sub(projected);
      velocity.copy(projected);
    }
    if (velocity.length() > maxSpeed) velocity.setLength(maxSpeed);
  }

  return velocity;
}

function computeBvcAvoidance(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 2.85;
  const tau = 0.45;
  const maxSpeed = drone.speed * 1.12;
  const constraints = [];
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const offset = other.position.clone().sub(drone.position);
    const distance = offset.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance < range) drone.neighborCount += 1;
    if (distance > range * 2.4) continue;

    const normal = offset.multiplyScalar(1 / distance);
    const buffer = Math.max(drone.radius + other.radius, (drone.safety + other.safety) * 0.5);
    const limit = (distance * 0.5 - buffer) / tau;
    constraints.push({ normal, limit });
  }

  if (!constraints.length) return nearest;
  const feasibleVelocity = projectBufferedVoronoiVelocity(drone.desired, constraints, maxSpeed);
  drone.avoidance.add(feasibleVelocity.sub(drone.desired));
  return nearest;
}

function buildDmpcPredictions(others, horizon, dt) {
  return others.map((other) => {
    const points = [];
    const pos = other.position.clone();
    const sequence = other.mpcPlan?.sequence;
    for (let step = 0; step < horizon; step += 1) {
      const planned = sequence?.[Math.min(step, sequence.length - 1)];
      const velocity = planned ?? other.plan?.vel ?? other.velocity;
      pos.addScaledVector(velocity, dt);
      points.push(pos.clone());
    }
    return { other, points, safety: other.safety };
  });
}

function makeDmpcAxes(desired, velocity, seed) {
  const forward = desired.lengthSq() > 0.0001 ? desired.clone().normalize() : velocity.clone();
  if (forward.lengthSq() < 0.0001) forward.set(1, 0, 0);
  forward.normalize();
  const side = perpendicularUnit(forward, seed);
  const lift = new THREE.Vector3().crossVectors(forward, side);
  if (lift.lengthSq() < 0.0001) lift.set(0, 1, 0);
  lift.normalize();
  return [forward, side, lift];
}

function seedDmpcSequence(drone, desired, horizon, maxSpeed) {
  const sequence = [];
  const previous = drone.mpcPlan?.sequence;
  for (let i = 0; i < horizon; i += 1) {
    const warm = previous?.[Math.min(i + 1, previous.length - 1)];
    sequence.push(clampVelocity(warm ?? desired, maxSpeed));
  }
  return sequence;
}

function evaluateDmpcSequence(drone, sequence, predictions, desired, dt, dsafe, maxSpeed, maxAccel) {
  const pos = drone.position.clone();
  const velocity = drone.velocity.clone();
  const prevAccel = new THREE.Vector3();
  const targetIndex = Math.min(drone.path.length - 1, drone.waypoint + 4);
  const targetPoint = drone.path[targetIndex] ?? drone.position;
  let cost = 0;

  for (let step = 0; step < sequence.length; step += 1) {
    const targetVelocity = clampVelocity(sequence[step], maxSpeed);
    const accel = targetVelocity.sub(velocity);
    const maxDelta = maxAccel * dt;
    if (accel.length() > maxDelta) accel.setLength(maxDelta);
    velocity.add(accel);
    if (velocity.length() > maxSpeed) velocity.setLength(maxSpeed);
    pos.addScaledVector(velocity, dt);

    const phase = (step + 1) / sequence.length;
    cost += velocity.distanceToSquared(desired) * (0.82 + phase * 0.28);
    cost += accel.lengthSq() * 0.34;
    cost += accel.clone().sub(prevAccel).lengthSq() * 0.12;
    cost += pos.distanceToSquared(targetPoint) * 0.026 * phase;
    prevAccel.copy(accel);

    for (const prediction of predictions) {
      const separation = dsafe + prediction.safety * 0.25;
      const d = pos.distanceTo(prediction.points[step]);
      if (d < separation) {
        const depth = separation - d;
        cost += depth * depth * 185 * (1.1 - step / (sequence.length + 3));
      } else if (d < separation * 1.55) {
        const margin = separation * 1.55 - d;
        cost += margin * margin * 2.8;
      }
    }

    const clearance = sampleEsdf(pos);
    if (clearance < 1.05) cost += (1.05 - clearance) * (1.05 - clearance) * 58;
    if (pos.y < 1.25) cost += (1.25 - pos.y) * (1.25 - pos.y) * 34;
    if (pos.y > 14.8) cost += (pos.y - 14.8) * (pos.y - 14.8) * 34;
  }

  return cost;
}

function optimizeDmpcSequence(drone, desired, predictions, horizon, dt, maxSpeed, maxAccel, passes) {
  let sequence = seedDmpcSequence(drone, desired, horizon, maxSpeed);
  let bestCost = evaluateDmpcSequence(drone, sequence, predictions, desired, dt, drone.safety * 1.42, maxSpeed, maxAccel);
  const axes = makeDmpcAxes(desired, drone.velocity, drone.id);
  const stepSizes = [maxSpeed * 0.42, maxSpeed * 0.22, maxSpeed * 0.11];

  for (let pass = 0; pass < passes; pass += 1) {
    const stepSize = stepSizes[Math.min(pass, stepSizes.length - 1)];
    for (let k = 0; k < horizon; k += 1) {
      for (const axis of axes) {
        for (const sign of [-1, 1]) {
          const trial = sequence.map((velocity) => velocity.clone());
          trial[k] = clampVelocity(trial[k].addScaledVector(axis, sign * stepSize), maxSpeed);
          const cost = evaluateDmpcSequence(drone, trial, predictions, desired, dt, drone.safety * 1.42, maxSpeed, maxAccel);
          if (cost < bestCost) {
            sequence = trial;
            bestCost = cost;
          }
        }
      }
    }
  }

  return sequence;
}

function computeDmpcAvoidance(drone, neighbors) {
  const profile = getProfile();
  const horizon = state.count > 300 ? 4 : state.count > 100 ? 5 : 7;
  const dt = state.count > 300 ? 0.32 : 0.28;
  const range = profile.avoidRange ?? 2.7;
  const maxSpeed = drone.speed * 1.12;
  const maxAccel = drone.speed * 2.1;
  const maxNeighbors = state.count > 300 ? 6 : state.count > 100 ? 8 : 12;
  const passes = state.count > 300 ? 1 : state.count > 100 ? 2 : 3;
  let nearest = Number.POSITIVE_INFINITY;
  const nearby = [];

  for (const other of neighbors) {
    if (other === drone) continue;
    const distanceSq = drone.position.distanceToSquared(other.position);
    if (distanceSq < 0.0001) continue;
    const distance = Math.sqrt(distanceSq);
    nearest = Math.min(nearest, distance);
    if (distance < range) drone.neighborCount += 1;
    if (distance < Math.max(range * 3.4, 10)) nearby.push({ other, distanceSq });
  }

  nearby.sort((a, b) => a.distanceSq - b.distanceSq);
  const others = nearby.slice(0, maxNeighbors).map((entry) => entry.other);
  const predictions = buildDmpcPredictions(others, horizon, dt);
  const desired = clampVelocity(drone.desired, maxSpeed);
  const sequence = optimizeDmpcSequence(drone, desired, predictions, horizon, dt, maxSpeed, maxAccel, passes);
  const firstControl = sequence[0].clone();

  drone.mpcPlan = {
    sequence: sequence.map((velocity) => velocity.clone()),
    dt,
    updatedAt: state.elapsed,
  };
  drone.policy.copy(firstControl);
  drone.avoidance.add(firstControl.sub(drone.desired));
  return nearest;
}

function makeMaderTrajectory(position, velocities, dt) {
  const points = [];
  const cursor = position.clone();
  for (const velocity of velocities) {
    cursor.addScaledVector(velocity, dt);
    points.push(cursor.clone());
  }
  return points;
}

function maderControlAt(commit, now) {
  if (!commit?.velocities?.length) return null;
  const index = clamp(Math.floor((now - commit.startTime) / commit.dt), 0, commit.velocities.length - 1);
  return commit.velocities[index].clone();
}

function maderPredictedPoint(other, step, dt, now) {
  const commit = other.maderCommit;
  if (commit?.points?.length && commit.expiresAt >= now) {
    const age = Math.max(0, now - commit.startTime);
    const offset = Math.floor(age / commit.dt);
    return commit.points[Math.min(offset + step, commit.points.length - 1)].clone();
  }
  const velocity = other.plan?.vel ?? other.velocity;
  return other.position.clone().addScaledVector(velocity, (step + 1) * dt);
}

function maderTrajectoryIsClear(drone, points, neighbors, dt, dsafe, now) {
  for (let step = 0; step < points.length; step += 1) {
    const point = points[step];
    if (sampleEsdf(point) < 0.72 || point.y < 1.18 || point.y > 15.0) return false;
    for (const other of neighbors) {
      if (other === drone) continue;
      if (drone.position.distanceToSquared(other.position) > 150) continue;
      const otherPoint = maderPredictedPoint(other, step, dt, now);
      const separation = dsafe + other.safety * 0.22;
      if (point.distanceToSquared(otherPoint) < separation * separation) return false;
    }
  }
  return true;
}

function scoreMaderTrajectory(drone, velocities, points, desired) {
  const target = drone.path[Math.min(drone.path.length - 1, drone.waypoint + 4)] ?? drone.position;
  let cost = 0;
  let previous = drone.velocity;
  for (let i = 0; i < velocities.length; i += 1) {
    const velocity = velocities[i];
    const phase = (i + 1) / velocities.length;
    cost += velocity.distanceToSquared(desired) * (0.7 + phase * 0.35);
    cost += velocity.clone().sub(previous).lengthSq() * 0.2;
    cost += points[i].distanceToSquared(target) * 0.022 * phase;
    const clearance = sampleEsdf(points[i]);
    if (clearance < 1.1) cost += (1.1 - clearance) * (1.1 - clearance) * 35;
    previous = velocity;
  }
  return cost;
}

function buildMaderVelocitySequence(drone, target, horizon, maxSpeed) {
  const sequence = [];
  for (let i = 0; i < horizon; i += 1) {
    const alpha = clamp((i + 1) / 3, 0, 1);
    const velocity = drone.velocity.clone().lerp(target, alpha);
    sequence.push(clampVelocity(velocity, maxSpeed));
  }
  return sequence;
}

function makeMaderCandidates(drone, horizon, maxSpeed) {
  const base = clampVelocity(drone.desired, maxSpeed);
  const options = [
    base.clone(),
    rotatedAroundY(base, 0.42),
    rotatedAroundY(base, -0.42),
    rotatedAroundY(base, 0.86),
    rotatedAroundY(base, -0.86),
    base.clone().add(new THREE.Vector3(0, drone.speed * 0.36, 0)),
    base.clone().add(new THREE.Vector3(0, -drone.speed * 0.36, 0)),
    base.clone().multiplyScalar(0.62),
    base.clone().multiplyScalar(0.28),
    new THREE.Vector3(0, 0, 0),
  ];
  const candidates = options.map((option) => buildMaderVelocitySequence(drone, clampVelocity(option, maxSpeed), horizon, maxSpeed));

  const previous = drone.maderCommit?.velocities;
  if (previous?.length) {
    const shifted = [];
    for (let i = 1; i < previous.length; i += 1) shifted.push(clampVelocity(previous[i], maxSpeed));
    while (shifted.length < horizon) shifted.push(base.clone());
    candidates.unshift(shifted.slice(0, horizon));
  }

  const side = perpendicularUnit(base.lengthSq() > 0.0001 ? base.clone().normalize() : new THREE.Vector3(1, 0, 0), drone.id);
  const detour = [];
  for (let i = 0; i < horizon; i += 1) {
    const blend = i < Math.ceil(horizon / 2) ? 0.48 : 0.18;
    detour.push(clampVelocity(base.clone().addScaledVector(side, maxSpeed * blend), maxSpeed));
  }
  candidates.push(detour);

  return candidates;
}

function commitMaderTrajectory(drone, velocities, dt, now) {
  const points = makeMaderTrajectory(drone.position, velocities, dt);
  drone.maderPreviousCommit = drone.maderCommit ?? null;
  drone.maderCommit = {
    velocities: velocities.map((velocity) => velocity.clone()),
    points,
    dt,
    startTime: now,
    expiresAt: now + dt * velocities.length,
    revision: (drone.maderCommit?.revision ?? 0) + 1,
  };
  return drone.maderCommit;
}

function brakeMaderCommit(drone, horizon, dt, now) {
  const velocities = [];
  for (let i = 0; i < horizon; i += 1) {
    const scale = Math.max(0, 0.58 - i * 0.09);
    velocities.push(drone.velocity.clone().multiplyScalar(scale));
  }
  return commitMaderTrajectory(drone, velocities, dt, now);
}

function computeMaderAvoidance(drone, neighbors) {
  const now = state.elapsed;
  const profile = getProfile();
  const horizon = state.count > 300 ? 5 : 7;
  const dt = 0.3;
  const maxSpeed = drone.speed * 1.1;
  const dsafe = drone.safety * 1.28;
  const range = profile.avoidRange ?? 2.65;
  const hardRange = drone.safety * 1.5;
  const away = new THREE.Vector3();
  const nearby = [];
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    away.copy(drone.position).sub(other.position);
    const distance = away.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance < range) drone.neighborCount += 1;
    if (distance < Math.max(range * 3.4, 10)) nearby.push(other);
    if (distance < hardRange) {
      drone.avoidance.addScaledVector(away.normalize(), ((hardRange - distance) / hardRange) * 3.4);
    }
  }

  const currentPoints = drone.maderCommit?.points;
  const currentValid =
    currentPoints &&
    drone.maderCommit.expiresAt > now &&
    maderTrajectoryIsClear(drone, currentPoints, nearby, dt, dsafe * 0.96, now);

  if (!currentValid) {
    const old = drone.maderPreviousCommit;
    if (old?.points?.length && old.expiresAt > now && maderTrajectoryIsClear(drone, old.points, nearby, dt, dsafe, now)) {
      drone.maderCommit = old;
    } else {
      brakeMaderCommit(drone, horizon, dt, now);
    }
  }

  if (!drone.maderNextReplan || now >= drone.maderNextReplan || drone.maderCommit.expiresAt - now < dt * 2) {
    const candidates = makeMaderCandidates(drone, horizon, maxSpeed);
    let best = null;
    let bestCost = Number.POSITIVE_INFINITY;
    for (const velocities of candidates) {
      const points = makeMaderTrajectory(drone.position, velocities, dt);
      if (!maderTrajectoryIsClear(drone, points, nearby, dt, dsafe, now)) continue;
      const cost = scoreMaderTrajectory(drone, velocities, points, drone.desired);
      if (cost < bestCost) {
        best = velocities;
        bestCost = cost;
      }
    }
    if (best) commitMaderTrajectory(drone, best, dt, now);
    else brakeMaderCommit(drone, horizon, dt, now);
    drone.maderNextReplan = now + 0.28 + (drone.id % 7) * 0.045;
  }

  const control = maderControlAt(drone.maderCommit, now) ?? drone.desired.clone();
  drone.plan = { vel: control.clone(), next: drone.maderNextReplan ?? now + dt };
  drone.policy.copy(control);
  drone.avoidance.add(control.sub(drone.desired));
  return nearest;
}

function egoNeighborPoint(other, step, dt, now) {
  const ego = other.egoPlan;
  if (ego?.points?.length && now - ego.updatedAt < 1.4) {
    return ego.points[Math.min(step + 1, ego.points.length - 1)].clone();
  }
  const mader = other.maderCommit;
  if (mader?.points?.length && mader.expiresAt >= now) {
    const offset = Math.max(0, Math.floor((now - mader.startTime) / mader.dt));
    return mader.points[Math.min(offset + step, mader.points.length - 1)].clone();
  }
  const mpc = other.mpcPlan?.sequence;
  if (mpc?.length) {
    const point = other.position.clone();
    for (let i = 0; i <= step; i += 1) point.addScaledVector(mpc[Math.min(i, mpc.length - 1)], dt);
    return point;
  }
  return other.position.clone().addScaledVector(other.velocity, (step + 1) * dt);
}

function seedEgoControlPoints(drone, horizon, dt) {
  const points = [drone.position.clone()];
  const previous = drone.egoPlan?.points;
  for (let i = 1; i <= horizon; i += 1) {
    const warm = previous?.[Math.min(i + 1, previous.length - 1)];
    if (warm && drone.position.distanceToSquared(warm) < 120) {
      points.push(warm.clone());
      continue;
    }
    const pathIndex = Math.min(drone.path.length - 1, drone.waypoint + i);
    const guide = drone.path[pathIndex] ?? drone.position.clone().addScaledVector(drone.desired, i * dt);
    const forward = drone.position.clone().addScaledVector(drone.desired, i * dt);
    points.push(forward.lerp(guide, 0.34));
  }
  return points;
}

function optimizeEgoControlPoints(drone, points, neighbors, horizon, dt, dsafe, margin, iterations) {
  const now = state.elapsed;
  const routeDir = drone.desired.lengthSq() > 0.0001 ? drone.desired.clone().normalize() : new THREE.Vector3(1, 0, 0);
  const gradient = new THREE.Vector3();
  const correction = new THREE.Vector3();

  for (let iter = 0; iter < iterations; iter += 1) {
    for (let i = 1; i < points.length; i += 1) {
      gradient.set(0, 0, 0);
      const point = points[i];
      const prev = points[i - 1];
      const next = points[Math.min(i + 1, points.length - 1)];
      const pathTarget = drone.path[Math.min(drone.path.length - 1, drone.waypoint + i)] ?? point;

      gradient.add(point.clone().multiplyScalar(2).sub(prev).sub(next).multiplyScalar(0.42));
      gradient.add(point.clone().sub(pathTarget).multiplyScalar(0.18));

      const clearance = sampleEsdf(point);
      if (clearance < margin) {
        const obstacleGrad = esdfGradient(point, correction);
        gradient.addScaledVector(obstacleGrad, -(margin - clearance) * 1.9);
      }

      for (const other of neighbors) {
        if (other === drone) continue;
        const otherPoint = egoNeighborPoint(other, i - 1, dt, now);
        correction.copy(point).sub(otherPoint);
        const distance = correction.length();
        if (distance < 0.001 || distance >= dsafe) continue;
        const away = correction.multiplyScalar(1 / distance);
        const tangent = away.clone().addScaledVector(routeDir, -away.dot(routeDir));
        if (tangent.lengthSq() > 0.0001) tangent.normalize();
        const strength = (dsafe - distance) / dsafe;
        gradient.addScaledVector(away, -strength * 2.6);
        gradient.addScaledVector(tangent, -strength * 1.1 * (drone.id % 2 === 0 ? 1 : -1));
      }

      point.addScaledVector(gradient, -0.18);
      point.y = clamp(point.y, 1.22, 14.9);
    }
  }
}

function computeEgoAvoidance(drone, neighbors) {
  const profile = getProfile();
  const horizon = state.count > 300 ? 4 : 6;
  const iterations = state.count > 300 ? 2 : state.count > 100 ? 3 : 5;
  const dt = 0.28;
  const dsafe = drone.safety * 1.34;
  const range = profile.avoidRange ?? 2.55;
  const margin = (profile.obstacleMargin ?? 2.9) * 0.58;
  const maxSpeed = drone.speed * 1.16;
  const nearby = [];
  let nearest = Number.POSITIVE_INFINITY;

  for (const other of neighbors) {
    if (other === drone) continue;
    const distance = drone.position.distanceTo(other.position);
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance < range) drone.neighborCount += 1;
    if (distance < Math.max(range * 3.5, 10)) nearby.push(other);
  }

  const points = seedEgoControlPoints(drone, horizon, dt);
  optimizeEgoControlPoints(drone, points, nearby, horizon, dt, dsafe, margin, iterations);
  drone.egoPlan = {
    points: points.map((point) => point.clone()),
    dt,
    updatedAt: state.elapsed,
  };

  const first = points[1]?.clone().sub(drone.position).multiplyScalar(1 / dt) ?? drone.desired.clone();
  if (first.length() > maxSpeed) first.setLength(maxSpeed);
  drone.policy.copy(first);
  drone.avoidance.add(first.sub(drone.desired));
  return nearest;
}

function computeOlfatiSaberForces(drone, neighbors) {
  const profile = getProfile();
  const eps = 0.1;
  const h = 0.25;
  const a = 1.6;
  const b = 5.5;
  const c = Math.abs(a - b) / Math.sqrt(4 * a * b);
  const range = profile.perceptionRadius ?? 6.6;
  const spacing = (profile.avoidRange ?? 3.25) * 0.82;
  const sigmaNorm = (z) => (Math.sqrt(1 + eps * z * z) - 1) / eps;
  const bump = (s) => (s < h ? 1 : s > 1 ? 0 : 0.5 * (1 + Math.cos((Math.PI * (s - h)) / (1 - h))));
  const sigma1 = (z) => z / Math.sqrt(1 + z * z);
  const phi = (z) => 0.5 * ((a + b) * sigma1(z + c) + (a - b));
  const rSigma = sigmaNorm(range);
  const dSigma = sigmaNorm(spacing);
  let nearest = Number.POSITIVE_INFINITY;
  let count = 0;
  const gradient = new THREE.Vector3();
  const consensus = new THREE.Vector3();
  const beta = new THREE.Vector3();
  const qij = new THREE.Vector3();
  const dv = new THREE.Vector3();

  for (const other of neighbors) {
    if (other === drone) continue;
    qij.copy(other.position).sub(drone.position);
    const distance = qij.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > range) continue;
    drone.neighborCount += 1;
    count += 1;
    const sig = sigmaNorm(distance);
    const adjacency = bump(sig / rSigma);
    const scale = (phi(sig - dSigma) * adjacency) / Math.sqrt(1 + eps * distance * distance);
    gradient.addScaledVector(qij, scale);
    dv.copy(other.velocity).sub(drone.velocity);
    consensus.addScaledVector(dv, adjacency);
    const shell = spacing * 0.72;
    if (distance < shell) {
      const urgency = (shell - distance) / shell;
      gradient.addScaledVector(qij, (-(urgency * urgency) * 9.5) / distance);
    }
  }

  for (const obstacle of state.obstacles) {
    const closest = closestPointOnObstacle(drone.position, obstacle);
    qij.copy(drone.position).sub(closest);
    const distance = qij.length();
    if (distance < 0.001 || distance > spacing * 1.35) continue;
    const sig = sigmaNorm(distance);
    const adjacency = bump(sig / dSigma);
    beta.addScaledVector(qij, (adjacency * (spacing * 1.35 - distance) * 1.6) / distance);
  }

  if (count > 1) {
    gradient.multiplyScalar(1 / Math.max(1, count * 0.35));
    consensus.multiplyScalar(1 / Math.max(1, count * 0.35));
  }
  if (gradient.length() > 5.8) gradient.setLength(5.8);
  const navigation = drone.desired.clone().sub(drone.velocity).multiplyScalar(0.34);
  drone.avoidance.addScaledVector(gradient, 1.15);
  drone.avoidance.addScaledVector(consensus, 0.82);
  drone.avoidance.addScaledVector(beta, 0.9);
  drone.avoidance.add(navigation);
  return nearest;
}

function computeVasarhelyiForces(drone, neighbors) {
  const profile = getProfile();
  const rRep = profile.avoidRange ?? 3.0;
  const pRep = 1.4;
  const rFrict = 4.1;
  const cFrict = 0.5;
  const vFrict = 0.32;
  const pFrict = 3.2;
  const aFrict = 2.1;
  const vFlock = drone.speed * 0.86;
  const cShill = 0.52;
  const braking = (r, aa, pp) => {
    const rp = r * pp;
    if (rp <= 0) return 0;
    if (rp <= aa / pp) return rp;
    return Math.sqrt(Math.max(0, 2 * aa * r - (aa * aa) / (pp * pp)));
  };
  let nearest = Number.POSITIVE_INFINITY;
  const rep = new THREE.Vector3();
  const frict = new THREE.Vector3();
  const offset = new THREE.Vector3();
  const dv = new THREE.Vector3();
  const heading = drone.velocity.lengthSq() > 0.0001 ? drone.velocity.clone().normalize() : drone.desired.clone().normalize();

  for (const other of neighbors) {
    if (other === drone) continue;
    offset.copy(drone.position).sub(other.position);
    const distance = offset.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > 8) continue;
    const direction = offset.clone().multiplyScalar(1 / distance);
    const frontWeight = clamp(0.42 + 0.58 * Math.max(0, -direction.dot(heading)), 0.42, 1);
    if (distance < rRep) {
      drone.neighborCount += 1;
      rep.addScaledVector(direction, pRep * (rRep - distance) * frontWeight);
    }
    dv.copy(other.velocity).sub(drone.velocity);
    const radialVelocity = Math.max(0, -dv.dot(direction));
    const vDiff = dv.length();
    const vMax = Math.max(vFrict, braking(distance - rFrict, aFrict, pFrict));
    if (vDiff + radialVelocity > vMax) {
      frict.addScaledVector(dv, (cFrict * (vDiff + radialVelocity - vMax) * frontWeight) / Math.max(vDiff, 0.001));
    }
  }

  const desiredCruise = drone.desired.lengthSq() > 0.0001 ? drone.desired.clone().setLength(vFlock) : new THREE.Vector3();
  const shill = desiredCruise.sub(drone.velocity).multiplyScalar(cShill);
  drone.avoidance.add(rep).add(frict).add(shill);
  return nearest;
}

const learningLayerCache = new Map();

function getLearningLayer(key, inputCount, outputCount, seed) {
  const cacheKey = `${key}:${inputCount}:${outputCount}:${seed}`;
  let layer = learningLayerCache.get(cacheKey);
  if (!layer) {
    const weights = new Float32Array(inputCount * outputCount);
    const bias = new Float32Array(outputCount);
    for (let o = 0; o < outputCount; o += 1) {
      bias[o] = Math.sin(seed * 0.37 + (o + 1) * 1.113) * 0.12;
      for (let i = 0; i < inputCount; i += 1) {
        const a = Math.sin((i + 1) * 12.9898 + (o + 1) * 78.233 + seed * 37.719);
        const b = Math.cos((i + 1) * 4.1414 + (o + 1) * 19.191 + seed * 11.17);
        weights[o * inputCount + i] = ((a + b * 0.5) * 0.42) / Math.sqrt(inputCount);
      }
    }
    layer = { weights, bias, inputCount, outputCount };
    learningLayerCache.set(cacheKey, layer);
  }
  return layer;
}

function runLearningLayer(input, layer, activation = Math.tanh) {
  const output = new Array(layer.outputCount);
  for (let o = 0; o < layer.outputCount; o += 1) {
    let sum = layer.bias[o];
    const base = o * layer.inputCount;
    for (let i = 0; i < layer.inputCount; i += 1) {
      sum += input[i] * layer.weights[base + i];
    }
    output[o] = activation ? activation(sum) : sum;
  }
  return output;
}

function runTinyPolicyNetwork(input, key, hiddenCount, outputCount, seed) {
  const first = getLearningLayer(`${key}:h1`, input.length, hiddenCount, seed);
  const hidden = runLearningLayer(input, first);
  const second = getLearningLayer(`${key}:h2`, hidden.length, hiddenCount, seed + 0.71);
  const hidden2 = runLearningLayer(hidden, second);
  const output = getLearningLayer(`${key}:out`, hidden2.length, outputCount, seed + 1.37);
  return runLearningLayer(hidden2, output);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function limitLearningVector(vector, maxLength) {
  if (vector.length() > maxLength) vector.setLength(maxLength);
  return vector;
}

function vectorFromLearningOutput(output, maxLength) {
  const vector = new THREE.Vector3(output[0] ?? 0, output[1] ?? 0, output[2] ?? 0);
  if (vector.length() > 1) vector.normalize();
  return vector.multiplyScalar(maxLength);
}

function collectLearningObservation(drone, neighbors, senseRange, k = 4) {
  const closest = [];
  let nearest = Number.POSITIVE_INFINITY;
  const rel = new THREE.Vector3();
  const relVel = new THREE.Vector3();
  for (const other of neighbors) {
    if (other === drone) continue;
    rel.copy(other.position).sub(drone.position);
    const distance = rel.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance < senseRange) drone.neighborCount += 1;
    closest.push({ other, distance });
  }
  closest.sort((left, right) => left.distance - right.distance);

  const desiredDir = drone.desired.lengthSq() > 0.0001 ? drone.desired.clone().normalize() : new THREE.Vector3(1, 0, 0);
  const velocityDir = drone.velocity.lengthSq() > 0.0001 ? drone.velocity.clone().normalize() : new THREE.Vector3();
  const clearance = sampleEsdf(drone.position);
  const speedScale = Math.max(drone.speed, 0.2);
  const features = [
    desiredDir.x,
    desiredDir.y,
    desiredDir.z,
    velocityDir.x,
    velocityDir.y,
    velocityDir.z,
    clamp(drone.velocity.length() / speedScale, 0, 1.8) - 0.9,
    clamp(clearance / 5, 0, 1) * 2 - 1,
    clamp((drone.position.y - 1.2) / 13.8, 0, 1) * 2 - 1,
  ];

  for (let i = 0; i < k; i += 1) {
    const entry = closest[i];
    if (!entry || entry.distance > senseRange * 1.8) {
      features.push(0, 0, 0, 0, 0, 0, 0, 0);
      continue;
    }
    rel.copy(entry.other.position).sub(drone.position);
    relVel.copy(entry.other.velocity).sub(drone.velocity);
    const distance = Math.max(entry.distance, 0.001);
    const closing = clamp((-relVel.dot(rel)) / (distance * Math.max(speedScale + (entry.other.speed ?? 1), 0.5)), -1, 1);
    features.push(
      clamp(rel.x / senseRange, -1, 1),
      clamp(rel.y / senseRange, -1, 1),
      clamp(rel.z / senseRange, -1, 1),
      clamp(relVel.x / 3, -1, 1),
      clamp(relVel.y / 3, -1, 1),
      clamp(relVel.z / 3, -1, 1),
      1 - clamp(distance / senseRange, 0, 1),
      closing,
    );
  }

  return { features, closest, nearest, desiredDir, clearance };
}

function learningSeparationVelocity(drone, closest, range, gain) {
  const result = new THREE.Vector3();
  const away = new THREE.Vector3();
  const relVel = new THREE.Vector3();
  for (const entry of closest) {
    if (entry.distance >= range) break;
    away.copy(drone.position).sub(entry.other.position);
    const distance = Math.max(entry.distance, 0.001);
    const urgency = (range - distance) / range;
    relVel.copy(entry.other.velocity).sub(drone.velocity);
    const closing = clamp(relVel.dot(away) / (distance * Math.max(drone.speed + (entry.other.speed ?? 1), 0.5)), 0, 1);
    result.addScaledVector(away.normalize(), (urgency * urgency + closing * 0.35) * gain);
  }
  return result;
}

function computeGlasPolicy(drone, neighbors) {
  const profile = getProfile();
  const radius = (profile.avoidRange ?? 2.35) * 2.1;
  const obs = collectLearningObservation(drone, neighbors, radius, 5);
  const output = runTinyPolicyNetwork(obs.features, "glas-policy", 14, 4, 1.01);
  const learned = obs.desiredDir
    .clone()
    .multiplyScalar(drone.speed * (0.78 + 0.16 * output[3]))
    .add(vectorFromLearningOutput(output, drone.speed * 0.62));

  const aggregate = new THREE.Vector3();
  const away = new THREE.Vector3();
  const relVel = new THREE.Vector3();
  for (let i = 0; i < Math.min(obs.closest.length, 5); i += 1) {
    const entry = obs.closest[i];
    if (entry.distance > radius) break;
    away.copy(drone.position).sub(entry.other.position);
    relVel.copy(entry.other.velocity).sub(drone.velocity);
    const distance = Math.max(entry.distance, 0.001);
    const proximity = 1 - distance / radius;
    const closing = clamp(relVel.dot(away) / (distance * Math.max(drone.speed + (entry.other.speed ?? 1), 0.5)), 0, 1);
    const attentionInput = [
      proximity,
      closing,
      away.x / radius,
      away.y / radius,
      away.z / radius,
      clamp(relVel.x / 3, -1, 1),
      clamp(relVel.y / 3, -1, 1),
      clamp(relVel.z / 3, -1, 1),
    ];
    const attention = sigmoid(runTinyPolicyNetwork(attentionInput, "glas-attention", 8, 1, 1.79)[0] * 2.2 + proximity * 3.2 + closing);
    aggregate.addScaledVector(away.normalize(), attention * proximity * proximity * drone.speed * 1.15);
    aggregate.addScaledVector(relVel, attention * proximity * 0.18);
  }

  const risk = Math.max(
    clamp(1 - obs.nearest / (drone.safety * 2.4), 0, 1),
    clamp(1 - obs.clearance / 1.45, 0, 1),
  );
  const lambda = clamp((risk - 0.15) / 0.85 + output[3] * 0.05, 0, 1) ** 2;
  const policy = learned.clone().add(aggregate);
  if (lambda > 0.001) {
    const backup = aggregate.clone().multiplyScalar(1.4).addScaledVector(drone.velocity, -0.38);
    const normal = new THREE.Vector3();
    esdfGradient(drone.position, normal);
    backup.addScaledVector(normal, clamp(1.35 - obs.clearance, 0, 1.35) * drone.speed * 1.1);
    policy.multiplyScalar(1 - lambda).addScaledVector(backup, lambda);
  }
  limitLearningVector(policy, drone.speed * 1.25);
  drone.policy.copy(policy);
  drone.avoidance.add(policy.clone().sub(drone.desired));
  return obs.nearest;
}

function scorePrimalAction(drone, obs, unit, actionIndex, bucket) {
  const target = drone.position.clone();
  if (unit) target.addScaledVector(unit, 1.25);
  if (unit && isWorldBlocked(target, 0.6)) return Number.NEGATIVE_INFINITY;

  const currentField = distanceFieldAt(drone.position);
  const nextField = distanceFieldAt(target);
  const progress = clamp((currentField - nextField) / 3.5, -1, 1);
  const clearance = clamp(sampleEsdf(target) / 4, 0, 1);
  const align = unit ? unit.dot(obs.desiredDir) : -0.08;
  const velocityDir = drone.velocity.lengthSq() > 0.0001 ? drone.velocity.clone().normalize() : obs.desiredDir;
  const inertia = unit ? unit.dot(velocityDir) : 0;
  let occupancy = 0;
  const predicted = new THREE.Vector3();
  for (let i = 0; i < Math.min(obs.closest.length, 8); i += 1) {
    const entry = obs.closest[i];
    predicted.copy(entry.other.position).addScaledVector(entry.other.velocity, 0.42);
    const gap = predicted.distanceTo(target);
    const safe = drone.safety + entry.other.safety + 0.35;
    if (gap < safe * 1.65) occupancy += ((safe * 1.65 - gap) / (safe * 1.65)) ** 2;
  }

  const input = [
    progress,
    align,
    clearance * 2 - 1,
    clamp(occupancy, 0, 3) / 3,
    inertia,
    unit ? 0 : 1,
    Math.sin(bucket * 0.7 + drone.id * 0.13),
    Math.cos(bucket * 0.51 + actionIndex * 0.37),
    unit?.x ?? 0,
    unit?.y ?? 0,
    unit?.z ?? 0,
  ];
  const learned = runTinyPolicyNetwork(input, "primal-action", 10, 1, 2.83 + actionIndex * 0.11)[0];
  return learned * 1.35 + progress * 1.6 + align * 0.72 + clearance * 0.55 + inertia * 0.18 - occupancy * 1.8 - (unit ? 0 : 0.18);
}

function computePrimalPolicy(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 2.25;
  const obs = collectLearningObservation(drone, neighbors, range, 4);
  if (drone.actionUntil === undefined) drone.actionUntil = -1;
  if (state.elapsed >= drone.actionUntil) {
    const bucket = Math.floor(state.elapsed / 0.24);
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestDir = null;
    for (let oi = 0; oi <= neighborOffsets.length; oi += 1) {
      const unit = oi < neighborOffsets.length ? offsetUnits[oi] : null;
      const score = scorePrimalAction(drone, obs, unit, oi, bucket);
      if (score > bestScore) {
        bestScore = score;
        bestDir = unit;
      }
    }
    if (!drone.actionDir) drone.actionDir = new THREE.Vector3();
    if (bestDir) drone.actionDir.copy(bestDir).multiplyScalar(drone.speed * 0.95);
    else drone.actionDir.set(0, 0, 0);
    drone.actionUntil = state.elapsed + 0.24;
  }
  if (drone.actionDir) {
    const policy = drone.actionDir.clone().add(learningSeparationVelocity(drone, obs.closest, range, drone.speed * 0.55));
    limitLearningVector(policy, drone.speed * 1.12);
    drone.desired.copy(policy);
    drone.policy.copy(policy);
  }
  return obs.nearest;
}

function neuralCbfParameters(input, key, seed) {
  const output = runTinyPolicyNetwork(input, key, 10, 2, seed);
  return {
    marginScale: 1.04 + 0.14 * ((output[0] + 1) * 0.5),
    alpha: 1.35 + 1.45 * ((output[1] + 1) * 0.5),
  };
}

function computeCbfFilter(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 2.55;
  const filtered = drone.desired.clone();
  const p = new THREE.Vector3();
  const relVel = new THREE.Vector3();
  let nearest = Number.POSITIVE_INFINITY;
  for (let pass = 0; pass < 2; pass += 1) {
    for (const other of neighbors) {
      if (other === drone) continue;
      p.copy(drone.position).sub(other.position);
      const distance = p.length();
      if (distance < 0.001) continue;
      if (pass === 0) {
        nearest = Math.min(nearest, distance);
        if (distance < range) drone.neighborCount += 1;
      }
      if (distance > 7) continue;
      relVel.copy(other.velocity).sub(drone.velocity);
      const dsBase = (drone.safety + other.safety) * 0.58;
      const input = [
        clamp(distance / 7, 0, 1),
        p.x / 7,
        p.y / 7,
        p.z / 7,
        clamp(relVel.x / 3, -1, 1),
        clamp(relVel.y / 3, -1, 1),
        clamp(relVel.z / 3, -1, 1),
        clamp((-relVel.dot(p)) / (distance * Math.max(drone.speed + (other.speed ?? 1), 0.5)), -1, 1),
      ];
      const cbf = neuralCbfParameters(input, "neural-cbf-pair", 3.71);
      const ds = dsBase * cbf.marginScale;
      const hval = distance * distance - ds * ds;
      const rhs = -cbf.alpha * hval + 2 * p.dot(other.velocity);
      const lhs = 2 * p.dot(filtered);
      if (lhs < rhs) {
        filtered.addScaledVector(p, (rhs - lhs) / (2 * distance * distance));
      }
    }

    for (const obstacle of state.obstacles) {
      const closest = closestPointOnObstacle(drone.position, obstacle);
      p.copy(drone.position).sub(closest);
      const distance = p.length();
      if (distance < 0.001 || distance > 4.7) continue;
      const input = [
        clamp(distance / 4.7, 0, 1),
        p.x / 4.7,
        p.y / 4.7,
        p.z / 4.7,
        clamp(drone.velocity.x / 3, -1, 1),
        clamp(drone.velocity.y / 3, -1, 1),
        clamp(drone.velocity.z / 3, -1, 1),
        clamp(sampleEsdf(drone.position) / 5, 0, 1) * 2 - 1,
      ];
      const cbf = neuralCbfParameters(input, "neural-cbf-obstacle", 4.19);
      const ds = 1.0 * cbf.marginScale;
      const hval = distance * distance - ds * ds;
      const rhs = -cbf.alpha * hval;
      const lhs = 2 * p.dot(filtered);
      if (lhs < rhs) {
        filtered.addScaledVector(p, (rhs - lhs) / (2 * distance * distance));
      }
    }
  }
  limitLearningVector(filtered, drone.speed * 1.28);
  drone.policy.copy(filtered);
  drone.avoidance.add(filtered.clone().sub(drone.desired));
  return nearest;
}

function rolloutSafetyCost(drone, option, rawAction, base, neighbors, range, obs) {
  let cost = option.distanceToSquared(rawAction) * 0.26 + option.distanceToSquared(base) * 0.14;
  const predicted = new THREE.Vector3();
  const otherPredicted = new THREE.Vector3();
  for (const horizon of [0.35, 0.75, 1.15, 1.55]) {
    predicted.copy(drone.position).addScaledVector(option, horizon);
    if (isWorldBlocked(predicted, 0.9)) cost += 90;
    for (const other of neighbors) {
      if (other === drone) continue;
      otherPredicted.copy(other.position).addScaledVector(other.velocity, horizon);
      const gap = predicted.distanceTo(otherPredicted);
      const safe = drone.safety + other.safety + 0.35;
      if (gap < safe * 1.7) cost += ((safe * 1.7 - gap) / safe) ** 2 * 18;
    }
  }
  const valueInput = [
    ...obs.features.slice(0, 13),
    clamp(option.x / Math.max(drone.speed, 0.2), -1, 1),
    clamp(option.y / Math.max(drone.speed, 0.2), -1, 1),
    clamp(option.z / Math.max(drone.speed, 0.2), -1, 1),
    clamp(option.length() / Math.max(drone.speed, 0.2), 0, 1.4) - 0.7,
    clamp(range - Math.min(obs.nearest, range), 0, range) / range,
  ];
  const value = runTinyPolicyNetwork(valueInput, "rl-safety-value", 12, 1, 5.29)[0];
  return cost - value * 1.75;
}

function computeRlSafetyLayerPolicy(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 2.4;
  const obs = collectLearningObservation(drone, neighbors, range, 5);
  const policyOut = runTinyPolicyNetwork(obs.features, "rl-policy", 16, 4, 4.83);
  const base = drone.desired.clone();
  const rawAction = obs.desiredDir
    .clone()
    .multiplyScalar(drone.speed * (0.82 + 0.24 * ((policyOut[3] + 1) * 0.5)))
    .add(vectorFromLearningOutput(policyOut, drone.speed * 0.58));
  limitLearningVector(rawAction, drone.speed * 1.18);

  const options = [
    rawAction.clone(),
    base.clone(),
    base.clone().multiplyScalar(0.62),
    rotatedAroundY(rawAction, 0.5),
    rotatedAroundY(rawAction, -0.5),
    rawAction.clone().add(new THREE.Vector3(0, drone.speed * 0.32, 0)),
    rawAction.clone().add(new THREE.Vector3(0, -drone.speed * 0.26, 0)),
    rawAction.clone().add(learningSeparationVelocity(drone, obs.closest, range, drone.speed * 0.62)),
  ];
  let best = options[0].clone();
  let bestCost = Number.POSITIVE_INFINITY;
  for (const option of options) {
    limitLearningVector(option, drone.speed * 1.12);
    const cost = rolloutSafetyCost(drone, option, rawAction, base, neighbors, range, obs);
    if (cost < bestCost) {
      bestCost = cost;
      best = option.clone();
    }
  }
  drone.policy.copy(best);
  drone.avoidance.add(best.clone().sub(drone.desired));
  return obs.nearest;
}

function computeE2eRlPolicy(drone, neighbors) {
  const profile = getProfile();
  const range = profile.avoidRange ?? 2.2;
  const obs = collectLearningObservation(drone, neighbors, range, 5);
  const output = runTinyPolicyNetwork(obs.features, "ctde-e2e-policy", 18, 4, 6.07);
  const throttle = 0.74 + 0.34 * ((output[3] + 1) * 0.5);
  const policy = obs.desiredDir
    .clone()
    .multiplyScalar(drone.speed * throttle)
    .add(vectorFromLearningOutput(output, drone.speed * 0.72));

  const guard = learningSeparationVelocity(drone, obs.closest, range, drone.speed * 0.72);
  if (guard.lengthSq() > 0.0001) {
    policy.add(guard);
    policy.addScaledVector(obs.desiredDir, -Math.min(guard.length() / Math.max(drone.speed, 0.2), 1) * drone.speed * 0.14);
  }
  limitLearningVector(policy, drone.speed * 1.18);
  drone.policy.copy(policy);
  drone.avoidance.add(policy.clone().sub(drone.desired));
  return obs.nearest;
}
function computeSwarmAvoidance(drone, neighbors) {
  if (isAlgorithm("C01")) return computeVelocityObstacleAvoidance(drone, neighbors, "orca");
  if (isAlgorithm("C02")) return computeVelocityObstacleAvoidance(drone, neighbors, "rvo");
  if (isAlgorithm("C08")) return computeVelocityObstacleAvoidance(drone, neighbors, "hrvo");
  if (isAlgorithm("C03")) return computeBvcAvoidance(drone, neighbors);
  if (isAlgorithm("C04")) return computeDmpcAvoidance(drone, neighbors);
  if (isAlgorithm("C05")) return computeMaderAvoidance(drone, neighbors);
  if (isAlgorithm("C06")) return computeEgoAvoidance(drone, neighbors);
  if (isAlgorithm("D01")) return computeApfForces(drone, neighbors);
  if (isAlgorithm("D02")) return computeBoidsForces(drone, neighbors);
  if (isAlgorithm("D03")) return computeOlfatiSaberForces(drone, neighbors);
  if (isAlgorithm("D04")) return computeVasarhelyiForces(drone, neighbors);
  if (isAlgorithm("D05")) return computeSocialForceInteractions(drone, neighbors);
  if (isAlgorithm("E01")) return computeGlasPolicy(drone, neighbors);
  if (isAlgorithm("E02")) return computePrimalPolicy(drone, neighbors);
  if (isAlgorithm("E03")) return computeCbfFilter(drone, neighbors);
  if (isAlgorithm("E04")) return computeRlSafetyLayerPolicy(drone, neighbors);
  if (isAlgorithm("E05")) return computeE2eRlPolicy(drone, neighbors);

  let nearest = Number.POSITIVE_INFINITY;
  const profile = getProfile();
  const separationRange =
    profile.avoidRange ?? (state.mode === "avoid" ? 2.45 : state.mode === "field" ? 3.05 : state.mode === "learning" ? 2.2 : 1.85);

  for (const other of neighbors) {
    if (other === drone) continue;
    tempVector.copy(drone.position).sub(other.position);
    const distance = tempVector.length();
    if (distance < 0.001) continue;
    nearest = Math.min(nearest, distance);
    if (distance > separationRange) continue;
    drone.neighborCount += 1;
    const urgency = (separationRange - distance) / separationRange;
    const closing = Math.max(0, other.velocity.clone().sub(drone.velocity).dot(tempVector.clone().normalize()) * -0.18);
    const weight = profile.avoidWeight ?? (state.mode === "avoid" ? 3.8 : state.mode === "learning" ? 4.2 : 2.6);
    drone.avoidance.add(tempVector.normalize().multiplyScalar((urgency * urgency + closing) * weight));
  }

  return nearest;
}

function computeObstacleAvoidance(drone) {
  if (isAlgorithm("D05")) {
    computeSocialObstacleForces(drone);
    return;
  }

  if (isAlgorithm("E03")) {
    if (drone.position.y < 1.45) drone.avoidance.y += (1.45 - drone.position.y) * 4;
    if (drone.position.y > 14.4) drone.avoidance.y -= (drone.position.y - 14.4) * 4;
    return;
  }

  const profile = getProfile();
  const margin = profile.obstacleMargin ?? (state.mode === "learning" ? 3.0 : 2.45);
  for (const obstacle of state.obstacles) {
    const closest = closestPointOnObstacle(drone.position, obstacle);
    tempVector.copy(drone.position).sub(closest);
    const distance = tempVector.length();
    if (distance < 0.001 || distance > margin) continue;
    const strength = (margin - distance) / margin;
    drone.avoidance.add(tempVector.normalize().multiplyScalar(strength * strength * 4.8));
  }

  if (drone.position.y < 1.45) {
    drone.avoidance.y += (1.45 - drone.position.y) * 4;
  }
  if (drone.position.y > 14.4) {
    drone.avoidance.y -= (drone.position.y - 14.4) * 4;
  }
}

function computeSocialObstacleForces(drone) {
  const profile = getProfile();
  const margin = profile.obstacleMargin ?? 4;
  const amplitude = (profile.socialA ?? 4.2) * 1.25;
  const decay = (profile.socialB ?? 0.9) * 0.92;

  for (const obstacle of state.obstacles) {
    const closest = closestPointOnObstacle(drone.position, obstacle);
    const offset = drone.position.clone().sub(closest);
    const distance = offset.length();
    if (distance < 0.001 || distance > margin) continue;
    const force = amplitude * Math.exp((drone.safety * 0.75 - distance) / decay);
    drone.avoidance.add(offset.normalize().multiplyScalar(force));
  }

  const floorForce = Math.exp((1.45 - drone.position.y) / decay);
  const ceilingForce = Math.exp((drone.position.y - 14.4) / decay);
  drone.avoidance.y += floorForce * 1.8;
  drone.avoidance.y -= ceilingForce * 1.8;
}

function closestPointOnObstacle(point, obstacle) {
  if (obstacle.kind === "box") {
    const s = obstacle.size;
    const c = obstacle.center;
    return new THREE.Vector3(
      clamp(point.x, c.x - s.x / 2, c.x + s.x / 2),
      clamp(point.y, c.y - s.y / 2, c.y + s.y / 2),
      clamp(point.z, c.z - s.z / 2, c.z + s.z / 2),
    );
  }
  if (obstacle.kind === "cylinder") {
    const horizontal = new THREE.Vector3(point.x - obstacle.center.x, 0, point.z - obstacle.center.z);
    if (horizontal.lengthSq() < 0.0001) horizontal.set(1, 0, 0);
    horizontal.normalize().multiplyScalar(obstacle.radius);
    return new THREE.Vector3(
      obstacle.center.x + horizontal.x,
      clamp(point.y, obstacle.center.y - obstacle.height / 2, obstacle.center.y + obstacle.height / 2),
      obstacle.center.z + horizontal.z,
    );
  }
  return obstacle.center.clone().add(point.clone().sub(obstacle.center).normalize().multiplyScalar(obstacle.radius));
}

function applyModeForces(drone, neighbors, dt) {
  if (state.mode === "central") {
    const lane = ((drone.id % 7) - 3) * 0.14;
    drone.avoidance.y += lane;
    if (drone.neighborCount > 2) {
      drone.avoidance.add(drone.velocity.clone().multiplyScalar(-0.35));
    }
  }

  if (state.mode === "learning" && isAlgorithm("E04")) {
    const predicted = drone.position.clone().add(drone.policy.clone().multiplyScalar(0.65));
    const blocked = isWorldBlocked(predicted, 1.05) || drone.neighborCount > 2;
    if (blocked) {
      drone.avoidance.multiplyScalar(1.35);
      drone.desired.lerp(drone.avoidance.clone().setLength(Math.max(1.2, drone.speed * 0.72)), 0.22);
    }
  }

  if (state.mode === "optimize") {
    drone.desired.lerp(drone.velocity.clone(), Math.min(0.16, dt * 1.4));
  }
}

// ===== Centralized coordination (F-family) =====

function droneScheduleKeys(drone, extraDelay, dt) {
  const keys = [];
  if (!drone.cells || drone.cells.length < 2 || drone.cells.length !== drone.rawPath.length) return keys;
  let time = drone.startDelay + extraDelay;
  const speed = Math.max(0.6, drone.speed * 0.88);
  for (let i = 0; i < drone.cells.length; i += 1) {
    if (i > 0) time += drone.rawPath[i].distanceTo(drone.rawPath[i - 1]) / speed;
    keys.push(drone.cells[i].index * 4096 + Math.min(4095, Math.floor(time / dt)));
  }
  return keys;
}

function scheduleWithReservations(order, pad, dt) {
  const reserved = new Set();
  for (const drone of order) {
    let shift = 0;
    for (; shift < 55; shift += 1) {
      const keys = droneScheduleKeys(drone, shift * dt, dt);
      let conflict = false;
      for (const key of keys) {
        for (let p = -pad; p <= pad && !conflict; p += 1) {
          if (reserved.has(key + p)) conflict = true;
        }
        if (conflict) break;
      }
      if (!conflict) break;
    }
    drone.startDelay += shift * dt;
    for (const key of droneScheduleKeys(drone, 0, dt)) {
      for (let p = -pad; p <= pad; p += 1) reserved.add(key + p);
    }
  }
}

function dcpReservationKeys(drone, extraDelay, dt) {
  const keys = [];
  if (!drone.cells || drone.cells.length < 2 || drone.cells.length !== drone.rawPath.length) return keys;
  let time = drone.startDelay + extraDelay;
  const speed = Math.max(0.55, drone.speed * 0.86);
  for (let i = 0; i < drone.cells.length; i += 1) {
    if (i > 0) time += drone.rawPath[i].distanceTo(drone.rawPath[i - 1]) / speed;
    const bucket = Math.min(8191, Math.floor(time / dt));
    const cell = drone.cells[i].index;
    keys.push({ kind: "v", cell, bucket });
    if (i > 0) {
      const prev = drone.cells[i - 1].index;
      const lo = Math.min(prev, cell);
      const hi = Math.max(prev, cell);
      keys.push({ kind: "e", cell: `${lo}:${hi}`, bucket });
    }
  }
  return keys;
}

function dcpReservationConflict(reserved, keys, pad) {
  for (const item of keys) {
    for (let p = -pad; p <= pad; p += 1) {
      const bucket = item.bucket + p;
      if (reserved.has(`${item.kind}:${item.cell}:${bucket}`)) return true;
    }
  }
  return false;
}

function dcpReserve(reserved, keys, pad, droneId) {
  for (const item of keys) {
    for (let p = -pad; p <= pad; p += 1) {
      const bucket = item.bucket + p;
      reserved.set(`${item.kind}:${item.cell}:${bucket}`, droneId);
    }
  }
}

function scheduleDcpStagger(dt) {
  const pad = state.count > 300 ? 0 : 1;
  const maxShift = state.count > 300 ? 42 : 86;
  const order = [...state.drones].sort((a, b) => {
    const lenA = a.cells?.length ?? 0;
    const lenB = b.cells?.length ?? 0;
    return a.startDelay - b.startDelay || lenB - lenA || a.id - b.id;
  });
  const reserved = new Map();

  for (const drone of order) {
    drone.dcpBaseDelay ??= drone.startDelay;
    drone.startDelay = drone.dcpBaseDelay;
    let selectedShift = 0;
    for (; selectedShift <= maxShift; selectedShift += 1) {
      const keys = dcpReservationKeys(drone, selectedShift * dt, dt);
      if (!dcpReservationConflict(reserved, keys, pad)) break;
    }
    drone.dcpDelay = selectedShift * dt;
    drone.startDelay = drone.dcpBaseDelay + drone.dcpDelay;
    dcpReserve(reserved, dcpReservationKeys(drone, 0, dt), pad, drone.id);
  }

  const conflictRounds = state.count > 300 ? 30 : 90;
  for (let round = 0; round < conflictRounds; round += 1) {
    const seen = new Map();
    let delayed = null;
    let delayedBucket = Number.POSITIVE_INFINITY;
    for (const drone of order) {
      for (const item of dcpReservationKeys(drone, 0, dt)) {
        const key = `${item.kind}:${item.cell}:${item.bucket}`;
        const other = seen.get(key);
        if (other !== undefined && other !== drone.id && item.bucket < delayedBucket) {
          delayed = drone;
          delayedBucket = item.bucket;
        } else {
          seen.set(key, drone.id);
        }
      }
    }
    if (!delayed) break;
    delayed.dcpDelay += dt;
    delayed.startDelay = delayed.dcpBaseDelay + delayed.dcpDelay;
  }
}
function scheduleCbsLite(dt) {
  const drones = state.drones;
  const rounds = state.count > 300 ? 140 : 340;
  for (let round = 0; round < rounds; round += 1) {
    const occupancy = new Map();
    let conflictPair = null;
    let conflictBucket = Number.POSITIVE_INFINITY;
    for (const drone of drones) {
      for (const key of droneScheduleKeys(drone, 0, dt)) {
        const existing = occupancy.get(key);
        if (existing !== undefined && existing !== drone.id) {
          const bucket = key % 4096;
          if (bucket < conflictBucket) {
            conflictBucket = bucket;
            conflictPair = [drones[existing], drone];
          }
        } else {
          occupancy.set(key, drone.id);
        }
      }
    }
    if (!conflictPair) break;
    const [a, b] = conflictPair;
    const lenA = a.cells ? a.cells.length : 0;
    const lenB = b.cells ? b.cells.length : 0;
    (lenA <= lenB ? a : b).startDelay += dt;
  }
}

function centralBaseDelay(drone) {
  if (drone.centralBaseDelay === undefined) drone.centralBaseDelay = drone.startDelay;
  return drone.centralBaseDelay;
}

function centralTimedKeys(drone, startDelay, dt) {
  const keys = [];
  if (!drone.cells || drone.cells.length < 2 || drone.cells.length !== drone.rawPath.length) return keys;
  let time = startDelay;
  const speed = Math.max(0.58, drone.speed * 0.88);
  for (let i = 0; i < drone.cells.length; i += 1) {
    if (i > 0) time += drone.rawPath[i].distanceTo(drone.rawPath[i - 1]) / speed;
    const bucket = Math.min(8191, Math.floor(time / dt));
    const cell = drone.cells[i].index;
    keys.push({ kind: "v", cell, bucket });
    if (i > 0) {
      const prev = drone.cells[i - 1].index;
      keys.push({ kind: "e", cell: `${prev}->${cell}`, reverse: `${cell}->${prev}`, bucket });
    }
  }
  return keys;
}

function centralKey(item, bucket = item.bucket, reverse = false) {
  if (item.kind === "e") return `${item.kind}:${reverse ? item.reverse : item.cell}:${bucket}`;
  return `${item.kind}:${item.cell}:${bucket}`;
}

function centralHasConflict(reserved, keys, pad) {
  for (const item of keys) {
    for (let p = -pad; p <= pad; p += 1) {
      const bucket = item.bucket + p;
      if (reserved.has(centralKey(item, bucket))) return true;
      if (item.kind === "e" && reserved.has(centralKey(item, bucket, true))) return true;
    }
  }
  return false;
}

function centralReserve(reserved, keys, pad, droneId) {
  for (const item of keys) {
    for (let p = -pad; p <= pad; p += 1) {
      const bucket = item.bucket + p;
      reserved.set(centralKey(item, bucket), droneId);
    }
  }
}

function centralScheduleByOrder(order, dt, pad, maxShift) {
  const reserved = new Map();
  const delays = new Map();
  for (const drone of order) {
    const base = centralBaseDelay(drone);
    let shift = 0;
    for (; shift <= maxShift; shift += 1) {
      const keys = centralTimedKeys(drone, base + shift * dt, dt);
      if (!centralHasConflict(reserved, keys, pad)) break;
    }
    const startDelay = base + shift * dt;
    delays.set(drone.id, startDelay);
    centralReserve(reserved, centralTimedKeys(drone, startDelay, dt), pad, drone.id);
  }
  return delays;
}

function centralFirstConflict(drones, delays, dt, pad) {
  const seen = new Map();
  for (const drone of drones) {
    const startDelay = delays.get(drone.id) ?? centralBaseDelay(drone);
    for (const item of centralTimedKeys(drone, startDelay, dt)) {
      for (let p = -pad; p <= pad; p += 1) {
        const bucket = item.bucket + p;
        const key = centralKey(item, bucket);
        const reverseKey = item.kind === "e" ? centralKey(item, bucket, true) : null;
        const other = seen.get(key) ?? (reverseKey ? seen.get(reverseKey) : undefined);
        if (other !== undefined && other !== drone.id) {
          return { a: other, b: drone.id, key, bucket, kind: item.kind };
        }
        seen.set(key, drone.id);
      }
    }
  }
  return null;
}

function centralCommitDelays(delays) {
  for (const drone of state.drones) {
    drone.startDelay = delays.get(drone.id) ?? centralBaseDelay(drone);
  }
}

function centralDelayCost(delays) {
  let cost = 0;
  for (const drone of state.drones) cost += Math.max(0, (delays.get(drone.id) ?? centralBaseDelay(drone)) - centralBaseDelay(drone));
  return cost;
}

function scheduleMapfReservations(dt) {
  const pad = state.count > 300 ? 0 : 1;
  const maxShift = state.count > 300 ? 70 : 130;
  const order = [...state.drones].sort((a, b) => centralBaseDelay(a) - centralBaseDelay(b) || a.id - b.id);
  const delays = centralScheduleByOrder(order, dt, pad, maxShift);
  centralCommitDelays(delays);
}

function scheduleCbsEcbs(dt) {
  const drones = state.drones;
  const pad = state.count > 300 ? 0 : 1;
  const maxExtra = (state.count > 300 ? 80 : 150) * dt;
  const nodeLimit = state.count > 300 ? 260 : 900;
  const root = { delays: new Map(drones.map((drone) => [drone.id, centralBaseDelay(drone)])), cost: 0, depth: 0 };
  const open = [root];
  let best = root;
  let bestConflictCount = Number.POSITIVE_INFINITY;

  for (let expanded = 0; expanded < nodeLimit && open.length; expanded += 1) {
    open.sort((a, b) => a.cost - b.cost || a.depth - b.depth);
    const bestCost = open[0].cost;
    const focalLimit = bestCost * 1.35 + dt * 2;
    let index = open.findIndex((node) => node.cost <= focalLimit && node.depth <= open[0].depth + 12);
    if (index < 0) index = 0;
    const [node] = open.splice(index, 1);
    const conflict = centralFirstConflict(drones, node.delays, dt, pad);
    if (!conflict) {
      centralCommitDelays(node.delays);
      return;
    }

    const conflictScore = node.depth;
    if (conflictScore < bestConflictCount) {
      bestConflictCount = conflictScore;
      best = node;
    }

    for (const id of [conflict.a, conflict.b]) {
      const childDelays = new Map(node.delays);
      const base = centralBaseDelay(drones[id]);
      const current = childDelays.get(id) ?? base;
      const delayed = current + dt;
      if (delayed - base > maxExtra) continue;
      childDelays.set(id, delayed);
      open.push({
        delays: childDelays,
        cost: centralDelayCost(childDelays) + node.depth * dt * 0.04,
        depth: node.depth + 1,
      });
    }
  }

  const fallbackOrder = [...drones].sort((a, b) => centralBaseDelay(a) - centralBaseDelay(b) || a.id - b.id);
  const fallback = centralScheduleByOrder(fallbackOrder, dt, pad, state.count > 300 ? 80 : 150);
  centralCommitDelays(bestConflictCount < Number.POSITIVE_INFINITY ? best.delays : fallback);
}

function topoPriorityOrder(drones, before) {
  const ids = drones.map((drone) => drone.id);
  const indegree = new Map(ids.map((id) => [id, 0]));
  for (const set of before.values()) for (const id of set) indegree.set(id, (indegree.get(id) ?? 0) + 1);
  const queue = ids
    .filter((id) => (indegree.get(id) ?? 0) === 0)
    .sort((a, b) => (drones[b].cells?.length ?? 0) - (drones[a].cells?.length ?? 0) || a - b);
  const order = [];
  while (queue.length) {
    const id = queue.shift();
    order.push(drones[id]);
    for (const next of before.get(id) ?? []) {
      indegree.set(next, (indegree.get(next) ?? 0) - 1);
      if ((indegree.get(next) ?? 0) === 0) {
        queue.push(next);
        queue.sort((a, b) => (drones[b].cells?.length ?? 0) - (drones[a].cells?.length ?? 0) || a - b);
      }
    }
  }
  if (order.length !== drones.length) return [...drones].sort((a, b) => (b.cells?.length ?? 0) - (a.cells?.length ?? 0) || a.id - b.id);
  return order;
}

function schedulePriorityBasedSearch(dt) {
  const drones = state.drones;
  const pad = state.count > 300 ? 0 : 1;
  const before = new Map();
  let order = [...drones].sort((a, b) => (b.cells?.length ?? 0) - (a.cells?.length ?? 0) || a.id - b.id);
  let bestDelays = null;
  const rounds = state.count > 300 ? 28 : 70;

  for (let round = 0; round < rounds; round += 1) {
    const delays = centralScheduleByOrder(order, dt, pad, state.count > 300 ? 70 : 130);
    bestDelays = delays;
    const conflict = centralFirstConflict(drones, delays, dt, pad);
    if (!conflict) break;
    const a = drones[conflict.a];
    const b = drones[conflict.b];
    const higher = (a.cells?.length ?? 0) >= (b.cells?.length ?? 0) ? a.id : b.id;
    const lower = higher === a.id ? b.id : a.id;
    if (!before.has(higher)) before.set(higher, new Set());
    before.get(higher).add(lower);
    order = topoPriorityOrder(drones, before);
  }

  centralCommitDelays(bestDelays ?? centralScheduleByOrder(order, dt, pad, state.count > 300 ? 70 : 130));
}
function polylineCumulative(points) {
  const cum = [0];
  for (let i = 1; i < points.length; i += 1) cum.push(cum[i - 1] + points[i].distanceTo(points[i - 1]));
  return cum;
}

function samplePolylineByDistance(points, cum, distance) {
  if (!points.length) return new THREE.Vector3();
  if (distance <= 0) return points[0].clone();
  const total = cum[cum.length - 1];
  if (distance >= total) return points[points.length - 1].clone();
  let seg = 1;
  while (seg < points.length - 1 && cum[seg] < distance) seg += 1;
  const span = Math.max(0.0001, cum[seg] - cum[seg - 1]);
  const alpha = clamp((distance - cum[seg - 1]) / span, 0, 1);
  return points[seg - 1].clone().lerp(points[seg], alpha);
}

function seedScpTrajectories(drones, steps, dtT) {
  return drones.map((drone) => {
    const path = drone.path.length > 1 ? drone.path : [drone.position.clone(), drone.position.clone().addScaledVector(drone.desired, 2)];
    const cum = polylineCumulative(path);
    const total = Math.max(cum[cum.length - 1], 0.001);
    const speed = Math.max(0.75, drone.speed * 0.9);
    const pts = [];
    const guide = [];
    for (let t = 0; t < steps; t += 1) {
      const dist = Math.min(total, t * dtT * speed);
      const point = samplePolylineByDistance(path, cum, dist);
      pts.push(point.clone());
      guide.push(point);
    }
    return { points: pts, guide };
  });
}

function applyScpLinearizedSeparation(trajs, step, dsep, cellSize, correctionScale) {
  const buckets = new Map();
  for (let i = 0; i < trajs.length; i += 1) {
    const p = trajs[i].points[step];
    const key = `${Math.floor(p.x / cellSize)},${Math.floor(p.y / cellSize)},${Math.floor(p.z / cellSize)}`;
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = [];
      buckets.set(key, bucket);
    }
    bucket.push(i);
  }

  const corrections = trajs.map(() => new THREE.Vector3());
  const checked = new Set();
  for (const [key, bucket] of buckets) {
    const [kx, ky, kz] = key.split(",").map(Number);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const otherBucket = buckets.get(`${kx + dx},${ky + dy},${kz + dz}`);
          if (!otherBucket) continue;
          for (const a of bucket) {
            for (const b of otherBucket) {
              if (a >= b) continue;
              const pairKey = `${a}:${b}`;
              if (checked.has(pairKey)) continue;
              checked.add(pairKey);
              const pa = trajs[a].points[step];
              const pb = trajs[b].points[step];
              const diff = pa.clone().sub(pb);
              const distance = diff.length();
              if (distance < 0.001 || distance > dsep * 1.55) continue;
              const normal = diff.multiplyScalar(1 / distance);
              const violation = dsep - normal.dot(pa.clone().sub(pb));
              if (violation <= 0) continue;
              corrections[a].addScaledVector(normal, violation * 0.5 * correctionScale);
              corrections[b].addScaledVector(normal, -violation * 0.5 * correctionScale);
            }
          }
        }
      }
    }
  }

  for (let i = 0; i < trajs.length; i += 1) trajs[i].points[step].add(corrections[i]);
}

function solveScpTrajectories(trajs, steps, iterations, dsep) {
  const gradient = new THREE.Vector3();
  const cellSize = Math.max(2.2, dsep);
  for (let iter = 0; iter < iterations; iter += 1) {
    for (let t = 1; t < steps - 1; t += 1) applyScpLinearizedSeparation(trajs, t, dsep, cellSize, 0.72);

    for (const traj of trajs) {
      for (let t = 1; t < steps - 1; t += 1) {
        const p = traj.points[t];
        const prev = traj.points[t - 1];
        const next = traj.points[t + 1];
        const guide = traj.guide[t];
        gradient.copy(p).multiplyScalar(2).sub(prev).sub(next);
        p.addScaledVector(gradient, -0.18);
        p.lerp(guide, 0.055);
        const clearance = sampleEsdf(p);
        if (clearance < 0.98) {
          esdfGradient(p, gradient);
          p.addScaledVector(gradient, (0.98 - clearance) * 0.62);
        }
        p.y = clamp(p.y, 1.2, 15);
      }
    }
  }
}

function scpDeconflict() {
  const drones = state.drones;
  if (drones.length < 2) return;
  const heavy = state.count > 300;
  const steps = heavy ? 32 : 48;
  const dtT = heavy ? 0.52 : 0.45;
  const iterations = heavy ? 8 : 18;
  const dsep = 1.92;
  const trajs = seedScpTrajectories(drones, steps, dtT);
  solveScpTrajectories(trajs, steps, iterations, dsep);

  for (const traj of trajs) {
    const gradient = new THREE.Vector3();
    for (let pass = 0; pass < 3; pass += 1) {
      for (const p of traj.points) {
        const clearance = sampleEsdf(p);
        if (clearance < 0.72) {
          esdfGradient(p, gradient);
          p.addScaledVector(gradient, 0.72 - clearance);
          p.y = clamp(p.y, 1.2, 15);
        }
      }
    }
  }

  drones.forEach((drone, i) => {
    drone.path = trajs[i].points.map((point) => point.clone());
    drone.rawPath = trajs[i].guide.map((point) => point.clone());
    drone.cells = null;
    drone.waypoint = 1;
    drone.speedScale = curvatureSpeedProfile(drone.path);
  });
}

function collectGoalCandidates(grid) {
  const candidates = [];
  for (let y = 1; y < grid.ny - 1; y += 1) {
    for (let z = 3; z < grid.nz - 3; z += 1) {
      const x = grid.nx - 3;
      const index = grid.index(x, y, z);
      if (grid.passable[index] && Number.isFinite(state.distance[index])) {
        candidates.push({ x, y, z, index, point: grid.cellToWorld(x, y, z) });
      }
    }
  }
  return candidates;
}

function assignmentCost(drone, goal) {
  const p = drone.position;
  const q = goal.point;
  return Math.hypot(q.x - p.x, (q.y - p.y) * 1.25, q.z - p.z);
}

function solveHungarianAssignment(drones, candidates) {
  const n = drones.length;
  const m = candidates.length;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1);
  const way = new Int32Array(m + 1);

  for (let i = 1; i <= n; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1);
    minv.fill(Number.POSITIVE_INFINITY);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Number.POSITIVE_INFINITY;
      let j1 = 0;
      for (let j = 1; j <= m; j += 1) {
        if (used[j]) continue;
        const cur = assignmentCost(drones[i0 - 1], candidates[j - 1]) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j += 1) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const assignment = new Int32Array(n).fill(-1);
  for (let j = 1; j <= m; j += 1) {
    if (p[j] > 0) assignment[p[j] - 1] = j - 1;
  }
  return assignment;
}

function solveAuctionAssignment(drones, candidates) {
  const m = candidates.length;
  const price = new Float64Array(m);
  const owner = new Int32Array(m).fill(-1);
  const assignment = new Int32Array(drones.length).fill(-1);
  const queue = drones.map((_, i) => i);
  const eps = 0.3;
  let guard = drones.length * Math.max(50, Math.min(400, candidates.length));
  while (queue.length && guard-- > 0) {
    const di = queue.shift();
    const drone = drones[di];
    let best = -1;
    let bestValue = -Infinity;
    let second = -Infinity;
    for (let j = 0; j < m; j += 1) {
      const value = -assignmentCost(drone, candidates[j]) - price[j];
      if (value > bestValue) {
        second = bestValue;
        bestValue = value;
        best = j;
      } else if (value > second) {
        second = value;
      }
    }
    if (best < 0) break;
    price[best] += bestValue - (Number.isFinite(second) ? second : bestValue) + eps;
    if (owner[best] >= 0) {
      assignment[owner[best]] = -1;
      queue.push(owner[best]);
    }
    owner[best] = di;
    assignment[di] = best;
  }

  for (let i = 0; i < drones.length; i += 1) {
    if (assignment[i] >= 0) continue;
    let best = 0;
    let bestCost = Number.POSITIVE_INFINITY;
    for (let j = 0; j < m; j += 1) {
      const cost = assignmentCost(drones[i], candidates[j]) + (owner[j] >= 0 ? 3 : 0);
      if (cost < bestCost) {
        bestCost = cost;
        best = j;
      }
    }
    assignment[i] = best;
  }
  return assignment;
}

function applyGoalAssignment(drones, candidates, assignment) {
  const grid = state.grid;
  const detailed = state.count <= 300;
  for (let i = 0; i < drones.length; i += 1) {
    const drone = drones[i];
    const goal = candidates[assignment[i]];
    if (!goal) continue;
    let route = null;
    if (detailed) {
      const startCell = nearestFreeCell(grid.worldToCell(drone.position), false);
      const cells = planGridAStar(startCell, new Set([goal.index]), goal);
      if (cells && cells.length > 1) route = routeFromCells(cells);
    }
    if (route) {
      drone.path = route.smooth.map((point) => point.clone());
      drone.rawPath = route.raw.map((point) => point.clone());
      drone.cells = route.cells;
    } else {
      const startPoint = drone.position.clone();
      const mid = startPoint.clone().lerp(goal.point, 0.5);
      let clear = false;
      for (let lift = 0; lift < 9; lift += 1) {
        if (!segmentBlocked(startPoint, mid, 0.5) && !segmentBlocked(mid, goal.point, 0.5)) {
          clear = true;
          break;
        }
        mid.y = Math.min(14.2, mid.y + 1.5);
      }
      if (clear) {
        const raw = [startPoint, mid, goal.point.clone()];
        drone.path = densifyPolyline(raw, 1.2);
        drone.rawPath = raw;
        drone.cells = null;
      } else {
        const fallback = tracePathFromCell(nearestFreeCell(grid.worldToCell(drone.position), true));
        drone.path = fallback.smooth.map((point) => point.clone());
        drone.rawPath = fallback.raw.map((point) => point.clone());
        drone.cells = fallback.cells;
      }
    }
    drone.assignmentGoal = goal.index;
    drone.waypoint = 1;
    drone.speedScale = null;
  }
}

function auctionAssign() {
  const grid = state.grid;
  const candidates = collectGoalCandidates(grid);
  if (!candidates.length) return;
  const drones = state.drones;
  if (candidates.length < drones.length) {
    const baseCount = candidates.length;
    for (let i = 0; candidates.length < drones.length; i += 1) candidates.push(candidates[i % baseCount]);
  }
  const useHungarian = drones.length <= 220 && candidates.length <= 360;
  const assignment = useHungarian ? solveHungarianAssignment(drones, candidates) : solveAuctionAssignment(drones, candidates);
  applyGoalAssignment(drones, candidates, assignment);
  scheduleMapfReservations(0.45);
}

function coordinateCentral() {
  const dt = 0.45;
  if (isAlgorithm("C07")) {
    scheduleDcpStagger(dt);
    return;
  }
  if (state.mode !== "central") return;
  if (isAlgorithm("F01")) scheduleMapfReservations(dt);
  else if (isAlgorithm("F02")) scheduleCbsEcbs(dt);
  else if (isAlgorithm("F03")) schedulePriorityBasedSearch(dt);
  else if (isAlgorithm("F04")) scpDeconflict();
  else if (isAlgorithm("F05")) auctionAssign();
}

function integrateDrone(drone, dt) {
  const profile = getProfile();
  const maxSpeed = state.mode === "avoid" ? drone.speed * 1.12 : drone.speed;
  const desired = drone.desired.clone().add(drone.avoidance);
  if (desired.length() > maxSpeed) desired.setLength(maxSpeed);
  const response = profile.response ?? (state.mode === "optimize" ? 2.2 : state.mode === "field" ? 2.8 : 3.5);
  drone.velocity.lerp(desired, Math.min(1, dt * response));
  if (drone.velocity.length() > maxSpeed) drone.velocity.setLength(maxSpeed);
  drone.position.add(drone.velocity.clone().multiplyScalar(dt));
  drone.position.y = clamp(drone.position.y, 1.15, 15.1);

  if (state.showTrails) {
    drone.trail.push(drone.position.clone());
    const maxTrail = state.count > 500 ? 4 : state.count > 100 ? 6 : 12;
    while (drone.trail.length > maxTrail) drone.trail.shift();
  } else {
    drone.trail.length = 0;
  }
}

function renderInstances() {
  const color = new THREE.Color();
  for (let i = 0; i < state.drones.length; i += 1) {
    const drone = state.drones[i];
    const direction = drone.velocity.lengthSq() > 0.0001 ? drone.velocity.clone().normalize() : forward;
    tempQuaternion.setFromUnitVectors(forward, direction);
    const scaleValue = i === state.selected ? 1.35 : 1;
    tempScale.set(scaleValue, scaleValue, scaleValue);
    tempMatrix.compose(drone.position, tempQuaternion, tempScale);
    droneMesh.setMatrixAt(i, tempMatrix);

    color.copy(drone.color);
    if (i === state.selected) color.lerp(new THREE.Color("#ffffff"), 0.58);
    droneMesh.setColorAt(i, color);

    const glowScale = i === state.selected ? 1.28 : 0.92;
    tempScale.set(glowScale, glowScale, glowScale);
    tempMatrix.compose(drone.position, identityQuaternion, tempScale);
    glowMesh.setMatrixAt(i, tempMatrix);
    glowMesh.setColorAt(i, i === state.selected ? agentHighlightColor : agentGlowColor);

    tempScale.set(drone.safety, drone.safety, drone.safety);
    tempMatrix.compose(drone.position, identityQuaternion, tempScale);
    safetyMesh.setMatrixAt(i, tempMatrix);
  }
  droneMesh.instanceMatrix.needsUpdate = true;
  droneMesh.instanceColor.needsUpdate = true;
  glowMesh.instanceMatrix.needsUpdate = true;
  glowMesh.instanceColor.needsUpdate = true;
  safetyMesh.instanceMatrix.needsUpdate = true;
  safetyMesh.visible = state.showSafety;

  const selected = state.drones[state.selected];
  if (selected) {
    selectedRing.position.copy(selected.position);
    selectedRing.scale.setScalar(1 + Math.sin(state.elapsed * 4.5) * 0.08);
  }
}

function rebuildTrailLines() {
  if (trailLines) {
    overlayGroup.remove(trailLines);
    trailLines.geometry.dispose();
    trailLines.material.dispose();
    trailLines = null;
  }
  const positions = [];
  const colors = [];
  if (state.showTrails) {
    for (const drone of state.drones) {
      for (let i = 0; i < drone.trail.length - 1; i += 1) {
        const a = drone.trail[i];
        const b = drone.trail[i + 1];
        positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
        const alpha = (i + 1) / Math.max(1, drone.trail.length - 1);
        const color = drone.color.clone().lerp(new THREE.Color("#ffffff"), alpha * 0.15);
        colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: state.count > 500 ? 0.18 : 0.34,
    depthWrite: false,
  });
  trailLines = new THREE.LineSegments(geometry, material);
  overlayGroup.add(trailLines);
}

function rebuildVectorLines() {
  if (vectorLines) {
    overlayGroup.remove(vectorLines);
    vectorLines.geometry.dispose();
    vectorLines.material.dispose();
    vectorLines = null;
  }

  const positions = [];
  const colors = [];
  if (state.showVectors) {
    const sampleEvery = state.count > 500 ? 36 : state.count > 100 ? 14 : state.count > 50 ? 6 : 1;
    for (const drone of state.drones) {
      if (drone.id !== state.selected && drone.id % sampleEvery !== 0) continue;
      pushVector(positions, colors, drone.position, drone.desired, new THREE.Color("#2ee6d6"), 0.74);
      pushVector(positions, colors, drone.position, drone.avoidance, new THREE.Color("#ffbf47"), 0.62);
      if (state.mode === "learning") {
        pushVector(positions, colors, drone.position, drone.policy, new THREE.Color("#b692ff"), 0.58);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.86,
    depthWrite: false,
  });
  vectorLines = new THREE.LineSegments(geometry, material);
  overlayGroup.add(vectorLines);
}

function pushVector(positions, colors, origin, vector, color, scale) {
  if (vector.lengthSq() < 0.015) return;
  const end = origin.clone().add(vector.clone().multiplyScalar(scale));
  positions.push(origin.x, origin.y, origin.z, end.x, end.y, end.z);
  colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
}

function updateSelectedReadout() {
  const drone = state.drones[state.selected];
  if (!drone) return;
  const algorithm = getAlgorithm();
  ui.selectedId.textContent = `#${String(drone.id + 1).padStart(3, "0")}`;
  ui.selectedMode.textContent = `${algorithm.id} ${algorithm.name}`;
  ui.selectedSpeed.textContent = `${drone.velocity.length().toFixed(1)} m/s`;
}

function updateStats() {
  ui.fps.textContent = Math.round(1000 / Math.max(1, state.frameMs)).toString();
  ui.conflict.textContent = state.conflicts.toString();
  ui.distance.textContent = state.averageSpacing ? `${state.averageSpacing.toFixed(1)} m` : "--";
  updateSelectedReadout();
}

function animate() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const start = performance.now();
  updateSimulation(dt);
  controls.update();
  renderInstances();
  if (state.showTrails && Math.floor(state.elapsed * 12) !== Math.floor((state.elapsed - dt) * 12)) {
    rebuildTrailLines();
  }
  if (state.showVectors && Math.floor(state.elapsed * 16) !== Math.floor((state.elapsed - dt) * 16)) {
    rebuildVectorLines();
  }
  renderer.render(scene, camera);
  state.frameMs = THREE.MathUtils.lerp(state.frameMs, performance.now() - start, 0.12);
  if (performance.now() - state.lastStatsAt > 220) {
    state.lastStatsAt = performance.now();
    updateStats();
  }
  requestAnimationFrame(animate);
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  if (width < 760) {
    camera.position.set(35, 32, 52);
    controls.target.set(0, 6, 0);
  }
}

function setActiveButton(selector, dataName, value) {
  document.querySelectorAll(selector).forEach((button) => {
    button.classList.toggle("active", button.dataset[dataName] === String(value));
  });
}

function syncFamilyButtons() {
  setActiveButton("[data-mode]", "mode", state.mode);
  const family = getFamily(state.mode);
  if (ui.algorithmFamilyLabel) {
    ui.algorithmFamilyLabel.textContent = family.shortTitle;
  }
}

function renderAlgorithmButtons() {
  if (!ui.algorithmButtons) return;
  const family = getFamily(state.mode);
  ui.algorithmButtons.replaceChildren();
  for (const algorithm of family.algorithms) {
    const fullAlgorithm = algorithmById.get(algorithm.id);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "algorithm-button";
    button.dataset.algorithm = algorithm.id;
    button.classList.toggle("active", algorithm.id === state.algorithmId);
    button.innerHTML = `
      <span class="algorithm-code">${algorithm.id}</span>
      <span class="algorithm-copy">
        <strong>${algorithm.name}</strong>
        <small>${algorithm.summary}</small>
      </span>
    `;
    button.addEventListener("click", () => {
      selectAlgorithm(fullAlgorithm.id);
    });
    ui.algorithmButtons.appendChild(button);
  }
  syncFamilyButtons();
}

function selectAlgorithm(algorithmId, regenerate = true) {
  const algorithm = algorithmById.get(algorithmId);
  if (!algorithm) return;
  const changedMode = state.mode !== algorithm.mode;
  state.algorithmId = algorithm.id;
  state.mode = algorithm.mode;
  syncFamilyButtons();
  renderAlgorithmButtons();
  renderAlgorithmInfo();
  if (changedMode || regenerate) {
    spawnDrones();
    updateModeVisuals();
    updateStats();
  }
}

function bindUi() {
  document.querySelectorAll("[data-count]").forEach((button) => {
    button.addEventListener("click", () => {
      state.count = Number(button.dataset.count);
      setActiveButton("[data-count]", "count", state.count);
      spawnDrones();
      updateStats();
    });
  });

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextMode = button.dataset.mode;
      selectAlgorithm(firstAlgorithmIdForMode(nextMode));
    });
  });

  renderAlgorithmButtons();

  ui.sceneSelect.addEventListener("change", () => {
    state.scenario = ui.sceneSelect.value;
    resetScene();
  });

  ui.pauseButton.addEventListener("click", () => {
    state.running = !state.running;
    ui.pauseIcon.textContent = state.running ? "||" : ">";
  });

  ui.resetButton.addEventListener("click", () => {
    resetScene();
  });

  const infoPanel = document.querySelector("#infoPanel");
  const infoToggle = document.querySelector("#infoToggle");
  if (infoPanel && infoToggle) {
    infoToggle.addEventListener("click", () => {
      infoPanel.classList.toggle("collapsed");
    });
    if (window.innerWidth < 1150) infoPanel.classList.add("collapsed");
  }

  document.querySelector("#togglePaths").addEventListener("change", (event) => {
    state.showPaths = event.target.checked;
    if (pathLines) pathLines.visible = state.showPaths;
  });
  document.querySelector("#toggleCorridor").addEventListener("change", (event) => {
    state.showCorridor = event.target.checked;
    rebuildSelectedVisuals();
  });
  document.querySelector("#toggleVectors").addEventListener("change", (event) => {
    state.showVectors = event.target.checked;
    rebuildVectorLines();
  });
  document.querySelector("#toggleSafety").addEventListener("change", (event) => {
    state.showSafety = event.target.checked;
    safetyMesh.visible = state.showSafety;
  });
  document.querySelector("#toggleGrid").addEventListener("change", (event) => {
    state.showGrid = event.target.checked;
    if (gridLines) gridLines.visible = state.showGrid;
    if (searchCloud) searchCloud.visible = state.mode === "search" || state.showGrid;
  });
  document.querySelector("#toggleTrails").addEventListener("change", (event) => {
    state.showTrails = event.target.checked;
    rebuildTrailLines();
  });

  canvas.addEventListener("pointerdown", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(droneMesh);
    if (hits.length && Number.isInteger(hits[0].instanceId)) {
      state.selected = hits[0].instanceId;
      rebuildSelectedVisuals();
      updateStats();
    }
  });
}

bindUi();
renderAlgorithmInfo();
resetScene();
resize();
window.addEventListener("resize", resize);
window.search3DLab = {
  getState: () => ({
    count: state.count,
    mode: state.mode,
    modeLabel: modeLabels[state.mode],
    algorithmId: state.algorithmId,
    algorithmName: getAlgorithm().name,
    drones: state.drones.length,
    conflicts: state.conflicts,
    averageSpacing: state.averageSpacing,
  }),
};
window.__search3dReady = true;
requestAnimationFrame(animate);



