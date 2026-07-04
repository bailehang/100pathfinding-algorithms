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
      { id: "C02", name: "RVO 3D", summary: "互惠速度障碍，局部避让直观", profile: { speed: 1.05, avoidRange: 2.35, avoidWeight: 3.7 } },
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
      { id: "E01", name: "GLAS", summary: "模仿学习分布式策略 + 安全滤波", profile: { speed: 1.05, avoidRange: 2.35, avoidWeight: 4.35 } },
      { id: "E02", name: "PRIMAL / PRIMAL2", summary: "RL + IL 多智能体路径策略", profile: { speed: 0.98, avoidRange: 2.25, avoidWeight: 4.1, wander: 0.12 } },
      { id: "E03", name: "Neural CBF", summary: "策略网络经控制屏障函数过滤", profile: { speed: 1.0, avoidRange: 2.55, avoidWeight: 4.7, obstacleMargin: 3.2 } },
      { id: "E04", name: "RL + Safety Layer", summary: "RL 输出经安全层修正", profile: { speed: 1.08, avoidRange: 2.4, avoidWeight: 4.55, wander: 0.1 } },
      { id: "E05", name: "End-to-End Swarm RL", summary: "端到端集群策略展示", profile: { speed: 1.12, avoidRange: 2.2, avoidWeight: 4.0, wander: 0.2 } },
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

  for (let i = 0; i < state.count; i += 1) {
    const start = startCellForDrone(i, rng);
    const route = tracePathFromCell(start);
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
      rawPath: route.raw,
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
  const hashCell = profile.avoidRange ? profile.avoidRange * 1.65 : state.mode === "field" ? 4.8 : 3.4;
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

function computeVelocityObstacleAvoidance(drone, neighbors, variant) {
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

function computeSwarmAvoidance(drone, neighbors) {
  if (isAlgorithm("C01")) return computeVelocityObstacleAvoidance(drone, neighbors, "orca");
  if (isAlgorithm("C08")) return computeVelocityObstacleAvoidance(drone, neighbors, "hrvo");
  if (isAlgorithm("D02", "D03", "D04")) return computeBoidsForces(drone, neighbors);
  if (isAlgorithm("D05")) return computeSocialForceInteractions(drone, neighbors);

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
