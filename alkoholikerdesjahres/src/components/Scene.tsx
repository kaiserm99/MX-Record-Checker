"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  ContactShadows,
  Environment,
  Float,
  Lightformer,
  Sparkles,
} from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

const GOLD = "#f2b233";
const GOLD_DEEP = "#b8811a";
const AMBER = "#ff8a1f";
const FOAM = "#fff5dc";

function useReducedMotionFlag() {
  const ref = useRef(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    ref.current = mq.matches;
    const onChange = () => (ref.current = mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return ref;
}

function GoldMaterial({ color = GOLD, roughness = 0.2 }: { color?: string; roughness?: number }) {
  return (
    <meshStandardMaterial
      color={color}
      metalness={1}
      roughness={roughness}
      envMapIntensity={1.6}
    />
  );
}

function Trophy() {
  const cupProfile = useMemo(() => {
    const pts: THREE.Vector2[] = [
      [0.02, 0.0],
      [0.3, 0.03],
      [0.48, 0.16],
      [0.62, 0.45],
      [0.7, 0.85],
      [0.74, 1.2],
      [0.8, 1.45],
      [0.86, 1.58],
      [0.84, 1.64],
      [0.76, 1.6],
      [0.7, 1.35],
    ].map(([x, y]) => new THREE.Vector2(x, y));
    return pts;
  }, []);

  return (
    <group position={[0, -0.55, 0]}>
      {/* cup */}
      <mesh castShadow>
        <latheGeometry args={[cupProfile, 72]} />
        <GoldMaterial />
      </mesh>
      {/* liquid */}
      <mesh position={[0, 1.34, 0]}>
        <cylinderGeometry args={[0.7, 0.7, 0.05, 48]} />
        <meshStandardMaterial color={AMBER} emissive={AMBER} emissiveIntensity={0.55} roughness={0.15} metalness={0.2} />
      </mesh>
      {/* foam */}
      <mesh position={[0, 1.58, 0]} scale={[1, 0.42, 1]}>
        <sphereGeometry args={[0.78, 40, 24, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color={FOAM} roughness={0.9} metalness={0} />
      </mesh>
      {/* handles */}
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * 1.0, 1.0, 0]} rotation={[0, 0, 0]}>
          <torusGeometry args={[0.42, 0.065, 20, 60]} />
          <GoldMaterial roughness={0.25} />
        </mesh>
      ))}
      {/* stem */}
      <mesh position={[0, -0.06, 0]}>
        <sphereGeometry args={[0.2, 32, 32]} />
        <GoldMaterial color={GOLD_DEEP} roughness={0.3} />
      </mesh>
      <mesh position={[0, -0.45, 0]}>
        <cylinderGeometry args={[0.13, 0.22, 0.7, 40]} />
        <GoldMaterial />
      </mesh>
      {/* base */}
      <mesh position={[0, -0.9, 0]}>
        <cylinderGeometry args={[0.62, 0.78, 0.22, 56]} />
        <GoldMaterial roughness={0.3} />
      </mesh>
      {/* plinth */}
      <mesh position={[0, -1.25, 0]} receiveShadow>
        <boxGeometry args={[1.9, 0.45, 1.9]} />
        <meshStandardMaterial color="#17111b" roughness={0.35} metalness={0.3} />
      </mesh>
      <mesh position={[0, -1.25, 0.96]}>
        <boxGeometry args={[1.1, 0.22, 0.02]} />
        <GoldMaterial roughness={0.35} />
      </mesh>
    </group>
  );
}

/** Deterministic PRNG so the scene renders identically on every mount. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function Bubbles({ count = 70 }: { count?: number }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const seeds = useMemo(() => {
    const rand = mulberry32(2026);
    return Array.from({ length: count }, () => ({
      x: (rand() - 0.5) * 7,
      y: rand() * 7 - 3.5,
      z: (rand() - 0.5) * 4 - 1,
      r: 0.03 + rand() * 0.08,
      v: 0.15 + rand() * 0.35,
      w: rand() * Math.PI * 2,
    }));
  }, [count]);
  const reduced = useReducedMotionFlag();

  useFrame((state, delta) => {
    if (!mesh.current) return;
    const t = state.clock.elapsedTime;
    seeds.forEach((s, i) => {
      if (!reduced.current) {
        s.y += s.v * delta;
        if (s.y > 3.8) s.y = -3.8;
      }
      dummy.position.set(s.x + Math.sin(t * 0.8 + s.w) * 0.12, s.y, s.z);
      dummy.scale.setScalar(s.r);
      dummy.updateMatrix();
      mesh.current!.setMatrixAt(i, dummy.matrix);
    });
    mesh.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 14, 14]} />
      <meshStandardMaterial
        color="#ffd97a"
        transparent
        opacity={0.28}
        roughness={0.1}
        metalness={0.6}
      />
    </instancedMesh>
  );
}

function BottleCap({
  position,
  color,
  speed,
  tilt,
}: {
  position: [number, number, number];
  color: string;
  speed: number;
  tilt: [number, number, number];
}) {
  return (
    <Float speed={speed} rotationIntensity={1.4} floatIntensity={1.6} floatingRange={[-0.25, 0.25]}>
      <group position={position} rotation={tilt}>
        <mesh>
          <cylinderGeometry args={[0.22, 0.22, 0.05, 22]} />
          <meshStandardMaterial color={color} metalness={0.85} roughness={0.3} emissive={color} emissiveIntensity={0.2} />
        </mesh>
        {[0.03, -0.03].map((y) => (
          <mesh key={y} position={[0, y, 0]}>
            <cylinderGeometry args={[0.15, 0.15, 0.01, 22]} />
            <meshStandardMaterial color={GOLD} metalness={1} roughness={0.2} emissive={GOLD_DEEP} emissiveIntensity={0.25} />
          </mesh>
        ))}
      </group>
    </Float>
  );
}

function Rig({ children }: { children: React.ReactNode }) {
  const group = useRef<THREE.Group>(null);
  const reduced = useReducedMotionFlag();
  const { pointer } = useThree();
  useFrame((state, delta) => {
    if (!group.current) return;
    const g = group.current;
    const targetY = pointer.x * 0.35 + (reduced.current ? 0 : state.clock.elapsedTime * 0.12);
    const targetX = -pointer.y * 0.18;
    g.rotation.y = THREE.MathUtils.damp(g.rotation.y, targetY, 2.2, delta);
    g.rotation.x = THREE.MathUtils.damp(g.rotation.x, targetX, 2.2, delta);
  });
  return <group ref={group}>{children}</group>;
}

function Lights() {
  return (
    <>
      <ambientLight intensity={0.35} />
      <spotLight position={[4, 6, 4]} angle={0.5} penumbra={0.8} intensity={60} color="#fff1cf" castShadow />
      <pointLight position={[-4, 1, 2]} intensity={25} color="#ff5a6e" />
      <pointLight position={[2, -2, -3]} intensity={18} color={AMBER} />
      <Environment resolution={128} frames={1}>
        <Lightformer form="ring" intensity={3} position={[0, 4, -3]} scale={5} color="#fff2cc" />
        <Lightformer form="rect" intensity={2} position={[-5, 1, 2]} scale={[3, 6, 1]} color="#ffc37a" />
        <Lightformer form="rect" intensity={1.2} position={[5, -1, 2]} scale={[3, 6, 1]} color="#ff7a8a" />
        <Lightformer form="circle" intensity={1.5} position={[0, -4, 3]} scale={4} color="#ffd97a" />
      </Environment>
    </>
  );
}

export default function Scene() {
  return (
    <Canvas
      dpr={[1, 1.6]}
      camera={{ position: [0, 0.4, 6.4], fov: 38 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      shadows
      style={{ background: "transparent" }}
    >
      <Lights />
      <Rig>
        <Float speed={1.4} rotationIntensity={0.25} floatIntensity={0.8} floatingRange={[-0.15, 0.15]}>
          <Trophy />
        </Float>
        <Bubbles />
        <BottleCap position={[-2.6, 1.6, -1]} color="#e0223f" speed={1.6} tilt={[1.1, 0.4, 0]} />
        <BottleCap position={[2.7, 1.9, -0.6]} color="#1b6f4a" speed={1.3} tilt={[0.6, 1.9, 0]} />
        <BottleCap position={[-2.2, -1.5, 0.4]} color="#f2b233" speed={1.8} tilt={[2.2, 0.8, 0]} />
        <BottleCap position={[2.4, -0.9, 0.8]} color="#2b2a7a" speed={1.1} tilt={[0.3, 2.6, 0]} />
        <BottleCap position={[0.4, 2.6, -2]} color="#e0223f" speed={1.5} tilt={[1.7, 1.2, 0]} />
        <Sparkles count={90} scale={[8, 6, 4]} size={2.4} speed={0.35} opacity={0.6} color="#ffd97a" />
      </Rig>
      <ContactShadows position={[0, -2.35, 0]} opacity={0.55} scale={9} blur={2.6} far={3} color="#000" />
    </Canvas>
  );
}
