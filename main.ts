const canvas = document.getElementById("myCanvas") as HTMLCanvasElement;
const c = canvas.getContext("2d") as CanvasRenderingContext2D;
canvas.width = window.innerWidth - 20;
canvas.height = window.innerHeight - 100;

canvas.focus();

let simHeight = 1.1;
let cScale = canvas.height / simHeight;
let simWidth = canvas.width / cScale;

let U_FIELD = 0;
let V_FIELD = 1;
let S_FIELD = 2;

let cnt = 0;

function cX(x: number) {
    return x * cScale;
}

function cY(y: number) {
    return canvas.height - y * cScale;
}

// ----------------- start of simulator ------------------------------

class Fluid {
    density: any;
    numX: number;
    numY: number;
    numCells: number;
    h: number;
    u: Float32Array;
    v: Float32Array;
    newU: Float32Array;
    newV: Float32Array;
    p: Float32Array;
    s: Float32Array;
    m: Float32Array;
    newM: Float32Array;

    constructor(density: number, numX: number, numY: number, h: number) {
        this.density = density;
        this.numX = numX + 2;
        this.numY = numY + 2;
        this.numCells = this.numX * this.numY;
        this.h = h;
        this.u = new Float32Array(this.numCells);
        this.v = new Float32Array(this.numCells);
        this.newU = new Float32Array(this.numCells);
        this.newV = new Float32Array(this.numCells);
        this.p = new Float32Array(this.numCells);
        this.s = new Float32Array(this.numCells);
        this.m = new Float32Array(this.numCells);
        this.newM = new Float32Array(this.numCells);
        this.m.fill(1.0);
    }

    integrate(dt: number, gravity: number) {
        const n = this.numY;

        for (let i = 1; i < this.numX; i++) {
            for (let j = 1; j < this.numY - 1; j++) {
                const idx = i * n + j;

                if (this.s[idx] === 0.0 || this.s[i * n + j - 1] === 0.0) {
                    continue;
                }

                this.v[idx] += gravity * dt;
            }
        }
    }

    solveIncompressibility(numIters: number, dt: number) {
        const n = this.numY;
        const cp = (this.density * this.h) / dt;

        for (let iter = 0; iter < numIters; iter++) {
            for (let i = 1; i < this.numX - 1; i++) {
                for (let j = 1; j < this.numY - 1; j++) {
                    const idx = i * n + j;

                    if (this.s[idx] === 0.0) {
                        continue;
                    }

                    let s = this.s[idx];
                    const sx0 = this.s[(i - 1) * n + j] as number;
                    const sx1 = this.s[(i + 1) * n + j] as number;
                    const sy0 = this.s[idx - 1] as number;
                    const sy1 = this.s[idx + 1] as number;
                    s = sx0 + sx1 + sy0 + sy1;

                    if (s == 0.0) {
                        continue;
                    }

                    const div = this.u[(i + 1) * n + j]! - this.u[idx]! + this.v[idx + 1]! - this.v[idx]!;
                    let p = -div / s;
                    p *= scene.overRelaxation;

                    this.p[idx] += cp * p;
                    this.u[idx] -= sx0 * p;
                    this.u[(i + 1) * n + j] += sx1 * p;
                    this.v[idx] -= sy0 * p;
                    this.v[idx + 1] += sy1 * p;
                }
            }
        }
    }

    extrapolate() {
        const n = this.numY;
        const lastRowIdx = (this.numX - 1) * n;

        for (let i = 0; i < this.numX; i++) {
            const idx = i * n;
            this.u[idx] = this.u[idx + 1]!;
            this.u[idx + this.numY - 1] = this.u[idx + this.numY - 2]!;
        }
        for (let j = 0; j < this.numY; j++) {
            this.v[j] = this.v[n + j]!;
            this.v[lastRowIdx + j] = this.v[lastRowIdx - n + j]!;
        }
    }

    sampleField(x: number, y: number, field: number) {
        const n = this.numY;
        const h = this.h;
        const h1 = 1.0 / h;
        const h2 = 0.5 * h;

        x = Math.max(Math.min(x, this.numX * h), h);
        y = Math.max(Math.min(y, this.numY * h), h);

        let dx = 0.0;
        let dy = 0.0;

        let f: Float32Array;

        switch (field) {
            case U_FIELD:
                f = this.u;
                dy = h2;
                break;
            case V_FIELD:
                f = this.v;
                dx = h2;
                break;
            case S_FIELD:
                f = this.m;
                dx = h2;
                dy = h2;
                break;
            default:
                f = new Float32Array([]);
                break;
        }

        const x0 = Math.min(Math.floor((x - dx) * h1), this.numX - 1);
        const tx = (x - dx - x0 * h) * h1;
        const x1 = Math.min(x0 + 1, this.numX - 1);

        const y0 = Math.min(Math.floor((y - dy) * h1), this.numY - 1);
        const ty = (y - dy - y0 * h) * h1;
        const y1 = Math.min(y0 + 1, this.numY - 1);

        const sx = 1.0 - tx;
        const sy = 1.0 - ty;

        const val = sx * sy * f[x0 * n + y0]! + tx * sy * f[x1 * n + y0]! + tx * ty * f[x1 * n + y1]! + sx * ty * f[x0 * n + y1]!;

        return val;
    }

    avgU(i: number, j: number) {
        const n = this.numY;
        const idx = i * n + j;
        const u = (this.u[idx - 1]! + this.u[idx]! + this.u[idx + n - 1]! + this.u[idx + n]!) * 0.25;
        return u;
    }

    avgV(i: number, j: number) {
        const n = this.numY;
        const idx = i * n + j;
        const prevRowIdx = (i - 1) * n;
        const v = (this.v[prevRowIdx + j]! + this.v[idx]! + this.v[prevRowIdx + j + 1]! + this.v[idx + 1]!) * 0.25;
        return v;
    }

    advectVel(dt: number) {
        this.newU.set(this.u);
        this.newV.set(this.v);

        let n = this.numY;
        let h = this.h;
        let h2 = 0.5 * h;

        for (let i = 1; i < this.numX; i++) {
            for (let j = 1; j < this.numY; j++) {
                cnt++;

                // u component
                const idx = i * n + j;
                if (this.s[idx] != 0.0 && this.s[(i - 1) * n + j] != 0.0 && j < this.numY - 1) {
                    let x = i * h;
                    let y = j * h + h2;
                    let u = this.u[idx] as number;
                    let v = this.avgV(i, j);

                    x = x - dt * u;
                    y = y - dt * v;
                    u = this.sampleField(x, y, U_FIELD);
                    this.newU[idx] = u;
                }
                // v component
                if (this.s[idx] != 0.0 && this.s[idx - 1] != 0.0 && i < this.numX - 1) {
                    let x = i * h + h2;
                    let y = j * h;
                    let u = this.avgU(i, j);
                    let v = this.v[idx] as number;

                    x = x - dt * u;
                    y = y - dt * v;
                    v = this.sampleField(x, y, V_FIELD);
                    this.newV[idx] = v;
                }
            }
        }

        this.u.set(this.newU);
        this.v.set(this.newV);
    }

    advectSmoke(dt: number) {
        this.newM.set(this.m);

        const n = this.numY;
        const h = this.h;
        const h2 = 0.5 * h;

        for (let i = 1; i < this.numX - 1; i++) {
            for (let j = 1; j < this.numY - 1; j++) {
                const idx = i * n + j;
                if (this.s[idx] === 0.0) continue;

                const u = (this.u[idx]! + this.u[(i + 1) * n + j]!) * 0.5;
                const v = (this.v[idx]! + this.v[idx + 1]!) * 0.5;
                const x = i * h + h2 - dt * u;
                const y = j * h + h2 - dt * v;

                this.newM[idx] = this.sampleField(x, y, S_FIELD);
            }
        }
        this.m.set(this.newM);
    }

    // ----------------- end of simulator ------------------------------

    simulate(dt: number, gravity: number, numIters: number) {
        this.integrate(dt, gravity);

        this.p.fill(0.0);
        this.solveIncompressibility(numIters, dt);

        this.extrapolate();
        this.advectVel(dt);
        this.advectSmoke(dt);
    }
}

let scene = {
    gravity: -9.81,
    dt: 1.0 / 120.0,
    numIters: 100,
    frameNr: 0,
    overRelaxation: 1.9,
    obstacleX: 0.0,
    obstacleY: 0.0,
    obstacleRadius: 0.15,
    paused: false,
    sceneNr: 0,
    showObstacle: false,
    showStreamlines: false,
    showVelocities: false,
    showPressure: false,
    showSmoke: true,
    fluid: null,
};

function setupScene(sceneNr = 0) {
    scene.sceneNr = sceneNr;
    scene.obstacleRadius = 0.15;
    scene.overRelaxation = 1.9;

    scene.dt = 1.0 / 60.0;
    scene.numIters = 40;

    let res = 100;

    if (sceneNr == 0) {
        res = 50;
    } else if (sceneNr == 3) {
        res = 200;
    }

    const domainHeight = 1.0;
    const domainWidth = (domainHeight / simHeight) * simWidth;
    const h = domainHeight / res;

    const numX = Math.floor(domainWidth / h);
    const numY = Math.floor(domainHeight / h);

    const density = 1000.0;

    let f = new Fluid(density, numX, numY, h);
    Object.defineProperty(scene, "fluid", {
        value: f,
        writable: true,
        enumerable: true,
        configurable: true,
    });

    const n = f.numY;

    if (sceneNr == 0) {
        // tank
        for (let i = 0; i < f.numX; i++) {
            for (let j = 0; j < f.numY; j++) {
                let s = 1.0; // fluid
                if (i == 0 || i == f.numX - 1 || j == 0) {
                    s = 0.0; // solid
                }
                f.s[i * n + j] = s;
            }
        }

        scene.gravity = -9.81;
        scene.showPressure = true;
        scene.showSmoke = false;
        scene.showStreamlines = false;
        scene.showVelocities = false;
    } else if (sceneNr == 1 || sceneNr == 3) {
        // vortex shedding
        const inVel = 2.0;
        for (let i = 0; i < f.numX; i++) {
            for (let j = 0; j < f.numY; j++) {
                let s = 1.0; // fluid
                const idx = i * n + j;

                if (i == 0 || j == 0 || j == f.numY - 1) {
                    s = 0.0; // solid
                }
                f.s[idx] = s;
                if (i == 1) {
                    f.u[idx] = inVel;
                }
            }
        }

        const pipeH = 0.1 * f.numY;
        const minJ = Math.floor(0.5 * f.numY - 0.5 * pipeH);
        const maxJ = Math.floor(0.5 * f.numY + 0.5 * pipeH);

        for (let j = minJ; j < maxJ; j++) {
            f.m[j] = 0.0;
        }

        setObstacle(0.4, 0.5, true);

        scene.gravity = 0.0;
        scene.showPressure = false;
        scene.showSmoke = true;
        scene.showStreamlines = false;
        scene.showVelocities = false;

        if (sceneNr == 3) {
            scene.dt = 1.0 / 120.0;
            scene.numIters = 100;
            scene.showPressure = true;
        }
    } else if (sceneNr == 2) {
        // paint
        scene.gravity = 0.0;
        scene.overRelaxation = 1.0;
        scene.showPressure = false;
        scene.showSmoke = true;
        scene.showStreamlines = false;
        scene.showVelocities = false;
        scene.obstacleRadius = 0.1;
    }

    (document.getElementById("streamButton") as HTMLInputElement).checked = scene.showStreamlines;
    (document.getElementById("velocityButton") as HTMLInputElement).checked = scene.showVelocities;
    (document.getElementById("pressureButton") as HTMLInputElement).checked = scene.showPressure;
    (document.getElementById("smokeButton") as HTMLInputElement).checked = scene.showSmoke;
    (document.getElementById("overrelaxButton") as HTMLInputElement).checked = scene.overRelaxation > 1.0;
}

// draw -------------------------------------------------------

function setColor(r: number, g: number, b: number) {
    c.fillStyle = `rgb(
${Math.floor(255 * r)},
${Math.floor(255 * g)},
${Math.floor(255 * b)})`;
    c.strokeStyle = `rgb(
${Math.floor(255 * r)},
${Math.floor(255 * g)},
${Math.floor(255 * b)})`;
}

function getSciColor(val: number, minVal: number, maxVal: number) {
    val = Math.min(Math.max(val, minVal), maxVal - 0.0001);
    const d = maxVal - minVal;
    val = d == 0.0 ? 0.5 : (val - minVal) / d;
    const m = 0.25;

    const num = Math.floor(val / m);
    const s = (val - num * m) / m;
    let r: number, g: number, b: number;

    switch (num) {
        case 0:
            r = 0.0;
            g = s;
            b = 1.0;
            break;
        case 1:
            r = 0.0;
            g = 1.0;
            b = 1.0 - s;
            break;
        case 2:
            r = s;
            g = 1.0;
            b = 0.0;
            break;
        case 3:
            r = 1.0;
            g = 1.0 - s;
            b = 0.0;
            break;
        default:
            r = 0.0;
            g = 0.0;
            b = 0.0;
            break;
    }

    return [255 * r, 255 * g, 255 * b, 255];
}

function draw() {
    c.clearRect(0, 0, canvas.width, canvas.height);

    c.fillStyle = "#FF0000";
    const f = scene.fluid;
    if (!f) return;

    let n = f.numY;
    const cellScale = 1.1;
    const h = f.h;

    let minP = f.p[0];
    let maxP = f.p[0];

    for (let i = 0; i < f.numCells; i++) {
        minP = Math.min(minP, f.p[i]);
        maxP = Math.max(maxP, f.p[i]);
    }

    const id = c.getImageData(0, 0, canvas.width, canvas.height);

    let color = [255, 255, 255, 255];

    for (let i = 0; i < f.numX; i++) {
        for (let j = 0; j < f.numY; j++) {
            const idx = i * n + j;
            if (scene.showPressure) {
                const p = f.p[idx];
                const s = f.m[idx];
                color = getSciColor(p, minP, maxP);
                if (scene.showSmoke) {
                    color[0] = Math.max(0.0, color[0]! - 255 * s);
                    color[1] = Math.max(0.0, color[1]! - 255 * s);
                    color[2] = Math.max(0.0, color[2]! - 255 * s);
                }
            } else if (scene.showSmoke) {
                const s = f.m[idx];
                color[0] = 255 * s;
                color[1] = 255 * s;
                color[2] = 255 * s;
                if (scene.sceneNr == 2) {
                    color = getSciColor(s, 0.0, 1.0);
                }
            } else if (f.s[idx] == 0.0) {
                color[0] = 0;
                color[1] = 0;
                color[2] = 0;
            }

            let x = Math.floor(cX(i * h));
            let y = Math.floor(cY((j + 1) * h));
            let cx = Math.floor(cScale * cellScale * h) + 1;
            let cy = Math.floor(cScale * cellScale * h) + 1;

            const r = color[0] ?? 255;
            const g = color[1] ?? 255;
            const b = color[2] ?? 255;

            for (let yi = y; yi < y + cy; yi++) {
                let p = 4 * (yi * canvas.width + x);

                for (let xi = 0; xi < cx; xi++) {
                    id.data[p++] = r;
                    id.data[p++] = g;
                    id.data[p++] = b;
                    id.data[p++] = 255;
                }
            }
        }
    }

    c.putImageData(id, 0, 0);

    if (scene.showVelocities) {
        c.strokeStyle = "#000000";
        const scale = 0.02;

        for (let i = 0; i < f.numX; i++) {
            for (let j = 0; j < f.numY; j++) {
                const u = f.u[i * n + j];
                const v = f.v[i * n + j];

                c.beginPath();

                const x0 = cX(i * h);
                const x1 = cX(i * h + u * scale);
                const y = cY((j + 0.5) * h);

                c.moveTo(x0, y);
                c.lineTo(x1, y);
                c.stroke();

                const x = cX((i + 0.5) * h);
                const y0 = cY(j * h);
                const y1 = cY(j * h + v * scale);

                c.beginPath();
                c.moveTo(x, y0);
                c.lineTo(x, y1);
                c.stroke();
            }
        }
    }

    if (scene.showStreamlines) {
        // var segLen = f.h * 0.2;
        const numSegs = 15;

        c.strokeStyle = "#000000";

        for (let i = 1; i < f.numX - 1; i += 5) {
            for (let j = 1; j < f.numY - 1; j += 5) {
                let x = (i + 0.5) * f.h;
                let y = (j + 0.5) * f.h;

                c.beginPath();
                c.moveTo(cX(x), cY(y));

                for (let n = 0; n < numSegs; n++) {
                    const u = f.sampleField(x, y, U_FIELD);
                    const v = f.sampleField(x, y, V_FIELD);
                    // const l = Math.sqrt(u * u + v * v);
                    // x += u/l * segLen;
                    // y += v/l * segLen;
                    x += u * 0.01;
                    y += v * 0.01;
                    if (x > f.numX * f.h) {
                        break;
                    }

                    c.lineTo(cX(x), cY(y));
                }
                c.stroke();
            }
        }
    }

    if (scene.showObstacle) {
        const r = scene.obstacleRadius + f.h;
        if (scene.showPressure) c.fillStyle = "#000000";
        else c.fillStyle = "#DDDDDD";
        c.beginPath();
        c.arc(cX(scene.obstacleX), cY(scene.obstacleY), cScale * r, 0.0, 2.0 * Math.PI);
        c.closePath();
        c.fill();

        c.lineWidth = 3.0;
        c.strokeStyle = "#000000";
        c.beginPath();
        c.arc(cX(scene.obstacleX), cY(scene.obstacleY), cScale * r, 0.0, 2.0 * Math.PI);
        c.closePath();
        c.stroke();
        c.lineWidth = 1.0;
    }

    if (scene.showPressure) {
        var s = "pressure: " + minP.toFixed(0) + " - " + maxP.toFixed(0) + " N/m";
        c.fillStyle = "#000000";
        c.font = "16px Arial";
        c.fillText(s, 10, 35);
    }
}

function setObstacle(x: number, y: number, reset: boolean) {
    let vx = 0.0;
    let vy = 0.0;

    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }

    scene.obstacleX = x;
    scene.obstacleY = y;
    const r = scene.obstacleRadius;
    const f = scene.fluid;
    const n = f.numY;

    for (let i = 1; i < f.numX - 2; i++) {
        for (let j = 1; j < f.numY - 2; j++) {
            const idx = i * n + j;

            f.s[idx] = 1.0;

            const dx = (i + 0.5) * f.h - x;
            const dy = (j + 0.5) * f.h - y;
            const distSq = dx * dx + dy * dy;

            if (distSq < r * r) {
                f.s[idx] = 0.0;
                if (scene.sceneNr === 2) {
                    f.m[idx] = 0.5 + 0.5 * Math.sin(0.1 * scene.frameNr);
                } else {
                    f.m[idx] = 1.0;
                }
                f.u[idx] = vx;
                f.u[idx + n] = vx;
                f.v[idx] = vy;
                f.v[idx + 1] = vy;
            }
        }
    }

    scene.showObstacle = true;
}

// interaction -------------------------------------------------------

let mouseDown = false;

function startDrag(x: number, y: number) {
    const bounds = canvas.getBoundingClientRect();
    const mx = x - bounds.left - canvas.clientLeft;
    const my = y - bounds.top - canvas.clientTop;
    mouseDown = true;

    x = mx / cScale;
    y = (canvas.height - my) / cScale;

    setObstacle(x, y, true);
}

function drag(x: number, y: number) {
    if (mouseDown) {
        const bounds = canvas.getBoundingClientRect();
        const mx = x - bounds.left - canvas.clientLeft;
        const my = y - bounds.top - canvas.clientTop;
        x = mx / cScale;
        y = (canvas.height - my) / cScale;
        setObstacle(x, y, false);
    }
}

function endDrag() {
    mouseDown = false;
}

canvas.addEventListener("mousedown", (event) => {
    startDrag(event.x, event.y);
});

canvas.addEventListener("mouseup", (event) => {
    endDrag();
});

canvas.addEventListener("mousemove", (event) => {
    drag(event.x, event.y);
});

canvas.addEventListener("touchstart", (event) => {
    if (event.touches.length <= 0) return;
    startDrag(event.touches[0]!.clientX, event.touches[0]!.clientY);
});

canvas.addEventListener("touchend", (event) => {
    endDrag();
});

canvas.addEventListener(
    "touchmove",
    (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (event.touches.length <= 0) return;
        drag(event.touches[0]!.clientX, event.touches[0]!.clientY);
    },
    { passive: false }
);

document.addEventListener("keydown", (event) => {
    switch (event.key) {
        case "p":
            scene.paused = !scene.paused;
            break;
        case "m":
            scene.paused = false;
            simulate();
            scene.paused = true;
            break;
    }
});

function toggleStart() {
    const button = document.getElementById("startButton") as HTMLButtonElement;
    if (scene.paused) {
        button.innerHTML = "Stop";
    } else {
        button.innerHTML = "Start";
    }
    scene.paused = !scene.paused;
}

// main -------------------------------------------------------

function simulate() {
    if (!scene.paused && scene.fluid !== null) {
        scene.fluid.simulate(scene.dt, scene.gravity, scene.numIters);
    }
    scene.frameNr++;
}

function update() {
    simulate();
    draw();
    requestAnimationFrame(update);
}

setupScene(1);
update();
