import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// ===================================================================
// CONFIGURATION GLOBALE
// ===================================================================
const CONFIG = {
    world: {
        size: 1000,
        gridSize: 100,
        particleCount: 20000
    },
    physics: {
        gravity: -9.8,
        moveSpeed: 40,
        sprintMultiplier: 2,
        jumpVelocity: 15
    },
    visuals: {
        bloom: true,
        fog: true,
        particles: true
    }
};

// ===================================================================
// SYSTÈME DE MÉTRIQUES
// ===================================================================
class MetricsSystem {
    constructor() {
        this.metrics = {
            activeAgents: 0,
            tasksPerSecond: 0,
            efficiency: 0,
            models: 0,
            datasets: 0,
            connections: 0
        };
        this.history = {
            tasks: [],
            efficiency: []
        };
        this.startTime = Date.now();
    }

    update() {
        document.getElementById('active-agents').textContent = this.metrics.activeAgents;
        document.getElementById('tasks-per-sec').textContent = this.metrics.tasksPerSecond.toFixed(1);
        document.getElementById('efficiency').textContent = `${Math.round(this.metrics.efficiency)}%`;
        document.getElementById('model-count').textContent = this.metrics.models;
        document.getElementById('dataset-count').textContent = this.metrics.datasets;
        document.getElementById('connection-count').textContent = this.metrics.connections;
    }

    addTask() {
        const now = Date.now();
        this.history.tasks.push(now);
        this.history.tasks = this.history.tasks.filter(t => now - t < 1000);
        this.metrics.tasksPerSecond = this.history.tasks.length;
    }
}

const metrics = new MetricsSystem();

// ===================================================================
// SCÈNE 3D AMÉLIORÉE
// ===================================================================
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000510, 0.002);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 10, 50);

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance"
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.5, 0.4, 0.85
);
composer.addPass(bloomPass);

// ===================================================================
// ÉCLAIRAGE DYNAMIQUE
// ===================================================================
const ambientLight = new THREE.AmbientLight(0x1a1a2e, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(50, 100, 50);
directionalLight.castShadow = true;
scene.add(directionalLight);

// ===================================================================
// CONTRÔLES AVANCÉS
// ===================================================================
const controls = new PointerLockControls(camera, document.body);
let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false, moveUp = false, moveDown = false, isSprinting = false;
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();

document.addEventListener('click', () => { if (!controls.isLocked) { controls.lock(); } });
document.addEventListener('keydown', (event) => {
    switch (event.code) {
        case 'KeyW': moveForward = true; break;
        case 'KeyS': moveBackward = true; break;
        case 'KeyA': moveLeft = true; break;
        case 'KeyD': moveRight = true; break;
        case 'Space': moveUp = true; break;
        case 'ShiftLeft': isSprinting = true; break;
    }
});
document.addEventListener('keyup', (event) => {
    switch (event.code) {
        case 'KeyW': moveForward = false; break;
        case 'KeyS': moveBackward = false; break;
        case 'KeyA': moveLeft = false; break;
        case 'KeyD': moveRight = false; break;
        case 'Space': moveUp = false; break;
        case 'ShiftLeft': isSprinting = false; break;
    }
});
document.addEventListener('contextmenu', (event) => {
    event.preventDefault();
    if (controls.isLocked) { showCreationMenu(); }
});

// ===================================================================
// MONDE PROCÉDURAL AMÉLIORÉ
// ===================================================================
const floorGeometry = new THREE.PlaneGeometry(CONFIG.world.size, CONFIG.world.size, CONFIG.world.gridSize, CONFIG.world.gridSize);
const floorMaterial = new THREE.MeshPhongMaterial({ color: 0x0a0a15, wireframe: true, emissive: 0x00aaff, emissiveIntensity: 0.02 });
const floor = new THREE.Mesh(floorGeometry, floorMaterial);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

// ===================================================================
// SYSTÈME D'ENTITÉS AMÉLIORÉ
// ===================================================================
const worldObjects = [];
const interactiveObjects = [];

class WorldEntity {
    constructor(name, type, position, color, size = 2) {
        this.name = name;
        this.type = type;
        this.position = position;
        this.color = new THREE.Color(color);
        this.size = size;
        this.data = { status: 'idle', version: '1.0.0', health: 100, performance: Math.random() * 100, connections: [], lastActivity: Date.now() };
        this.createMesh();
        worldObjects.push(this);
        interactiveObjects.push(this.mesh);
        scene.add(this.group);
        if (type === 'Model') metrics.metrics.models++;
        else if (type === 'Dataset') metrics.metrics.datasets++;
        else if (type === 'Agent') metrics.metrics.activeAgents++;
    }
    createMesh() {
        this.group = new THREE.Group();
        this.group.position.copy(this.position);
        let geometry, material;
        switch (this.type) {
            case 'Model':
                geometry = new THREE.IcosahedronGeometry(this.size, 2);
                material = new THREE.MeshPhongMaterial({ color: this.color, emissive: this.color, emissiveIntensity: 0.3, shininess: 100 });
                break;
            case 'Dataset':
                geometry = new THREE.BoxGeometry(this.size, this.size, this.size);
                material = new THREE.MeshPhongMaterial({ color: this.color, transparent: true, opacity: 0.8, emissive: this.color, emissiveIntensity: 0.1 });
                break;
            case 'Agent':
                geometry = new THREE.SphereGeometry(this.size * 0.5, 32, 32);
                material = new THREE.MeshPhongMaterial({ color: this.color, emissive: this.color, emissiveIntensity: 0.5 });
                break;
        }
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
        this.mesh.userData = { entity: this };
        this.group.add(this.mesh);
    }
    update(deltaTime) {
        if (this.type === 'Model' || this.type === 'Dataset') { this.mesh.rotation.y += deltaTime * 0.5; }
    }
    setStatus(status) {
        this.data.status = status;
        this.data.lastActivity = Date.now();
        this.mesh.material.emissiveIntensity = (status === 'working') ? 0.8 : 0.3;
    }
}

// ===================================================================
// SYSTÈME D'ÉVÉNEMENTS AMÉLIORÉ
// ===================================================================
class EventSystem {
    showNotification(message, type = 'info') {
        const notifContainer = document.getElementById('notifications');
        const notif = document.createElement('div');
        notif.className = `notification ${type}`;
        notif.textContent = message;
        notifContainer.appendChild(notif);
        setTimeout(() => {
            notif.style.opacity = '0';
            setTimeout(() => notif.remove(), 300);
        }, 3000);
    }
}
const eventSystem = new EventSystem();

// ===================================================================
// SYSTÈME DE CRÉATION D'ENTITÉS
// ===================================================================
function showCreationMenu() { document.getElementById('creation-menu').style.display = 'block'; controls.unlock(); }
function closeCreationMenu() { document.getElementById('creation-menu').style.display = 'none'; controls.lock(); }
window.createNewElement = function (type) {
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    const position = camera.position.clone().add(direction.multiplyScalar(20));
    position.y = 5;
    let entity;
    switch (type) {
        case 'agent': entity = new WorldEntity(`Agent-${Date.now().toString(36)}`, 'Agent', position, 0xffffff, 2); break;
        case 'model': entity = new WorldEntity(`Model-${Date.now().toString(36)}`, 'Model', position, 0x00ff00, 5); break;
        case 'dataset': entity = new WorldEntity(`Dataset-${Date.now().toString(36)}`, 'Dataset', position, 0x0088ff, 3); break;
    }
    eventSystem.showNotification(`🤖 Nouveau ${type} créé`, 'success');
    closeCreationMenu();
    metrics.update();
};
window.closeCreationMenu = closeCreationMenu;

// ===================================================================
// BOUCLE D'ANIMATION PRINCIPALE
// ===================================================================
const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    if (controls.isLocked) {
        const speed = CONFIG.physics.moveSpeed * (isSprinting ? CONFIG.physics.sprintMultiplier : 1);
        velocity.x -= velocity.x * 10.0 * deltaTime;
        velocity.z -= velocity.z * 10.0 * deltaTime;
        direction.z = Number(moveForward) - Number(moveBackward);
        direction.x = Number(moveRight) - Number(moveLeft);
        direction.normalize();
        if (moveForward || moveBackward) velocity.z -= direction.z * speed * deltaTime;
        if (moveLeft || moveRight) velocity.x -= direction.x * speed * deltaTime;
        controls.moveRight(-velocity.x * deltaTime);
        controls.moveForward(-velocity.z * deltaTime);
        camera.position.y += (Number(moveUp) - Number(moveDown)) * speed * deltaTime;
    }
    worldObjects.forEach(entity => entity.update(deltaTime));
    composer.render();
}
animate();

// ===================================================================
// GESTION DU REDIMENSIONNEMENT
// ===================================================================
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
});

// ===================================================================
// INITIALISATION FINALE
// ===================================================================
console.log('🌌 KaggleForge Latent Space Sandbox initialized!');
metrics.update();