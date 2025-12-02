'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

interface NodeProps {
  position: [number, number, number];
  color: string;
  scale?: number;
  pulseSpeed?: number;
}

function Node({ position, color, scale = 1, pulseSpeed = 1 }: NodeProps) {
  const ref = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ref.current) {
      ref.current.scale.setScalar(
        scale * (1 + Math.sin(state.clock.elapsedTime * pulseSpeed) * 0.1)
      );
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.1} floatIntensity={0.2}>
      <Sphere ref={ref} position={position} args={[0.1, 32, 32]}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          roughness={0.4}
          metalness={0.6}
        />
      </Sphere>
      {/* Glow effect */}
      <Sphere position={position} args={[0.13, 16, 16]}>
        <meshBasicMaterial color={color} transparent opacity={0.1} />
      </Sphere>
    </Float>
  );
}

interface ConnectionProps {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
}

function Connection({ start, end, color }: ConnectionProps) {
  return (
    <Line
      points={[start, end]}
      color={color}
      lineWidth={1}
      transparent
      opacity={0.4}
    />
  );
}

function DataParticle({ path, speed }: { path: [number, number, number][]; speed: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const progress = useRef(0);

  useFrame((state, delta) => {
    if (ref.current && path.length >= 2) {
      progress.current = (progress.current + delta * speed) % 1;

      const segmentLength = 1 / (path.length - 1);
      const segmentIndex = Math.floor(progress.current / segmentLength);
      const segmentProgress = (progress.current % segmentLength) / segmentLength;

      const startIndex = Math.min(segmentIndex, path.length - 2);
      const endIndex = startIndex + 1;

      ref.current.position.lerpVectors(
        new THREE.Vector3(...path[startIndex]),
        new THREE.Vector3(...path[endIndex]),
        segmentProgress
      );
    }
  });

  return (
    <Sphere ref={ref} args={[0.03, 8, 8]}>
      <meshBasicMaterial color="#52525b" />
    </Sphere>
  );
}

function NeuralNetworkScene() {
  // Network architecture: 4 -> 6 -> 6 -> 4 -> 1
  const layers = useMemo(() => {
    const layerSizes = [4, 6, 6, 4, 1];
    const layerSpacing = 1.2;
    const nodeSpacing = 0.6;

    return layerSizes.map((size, layerIndex) => {
      const x = (layerIndex - (layerSizes.length - 1) / 2) * layerSpacing;
      return Array.from({ length: size }, (_, nodeIndex) => {
        const y = (nodeIndex - (size - 1) / 2) * nodeSpacing;
        return [x, y, 0] as [number, number, number];
      });
    });
  }, []);

  const connections = useMemo(() => {
    const conns: { start: [number, number, number]; end: [number, number, number] }[] = [];

    for (let i = 0; i < layers.length - 1; i++) {
      for (const startNode of layers[i]) {
        for (const endNode of layers[i + 1]) {
          conns.push({ start: startNode, end: endNode });
        }
      }
    }

    return conns;
  }, [layers]);

  const dataPaths = useMemo(() => {
    const paths: [number, number, number][][] = [];

    for (let i = 0; i < 5; i++) {
      const path: [number, number, number][] = [];
      let currentLayer = 0;
      let currentNode = Math.floor(Math.random() * layers[0].length);
      path.push(layers[0][currentNode]);

      while (currentLayer < layers.length - 1) {
        currentLayer++;
        currentNode = Math.floor(Math.random() * layers[currentLayer].length);
        path.push(layers[currentLayer][currentNode]);
      }

      paths.push(path);
    }

    return paths;
  }, [layers]);

  const colors = ['#41414b', '#5d5d6c', '#737384', '#796c5d', '#918371'];

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#5d5d6c" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#796c5d" />

      {/* Connections */}
      {connections.map((conn, i) => (
        <Connection
          key={`conn-${i}`}
          start={conn.start}
          end={conn.end}
          color="#71717a"
        />
      ))}

      {/* Nodes */}
      {layers.map((layer, layerIndex) =>
        layer.map((position, nodeIndex) => (
          <Node
            key={`node-${layerIndex}-${nodeIndex}`}
            position={position}
            color={colors[layerIndex % colors.length]}
            scale={layerIndex === 0 || layerIndex === layers.length - 1 ? 1.2 : 1}
            pulseSpeed={1 + layerIndex * 0.2}
          />
        ))
      )}

      {/* Data particles flowing through */}
      {dataPaths.map((path, i) => (
        <DataParticle key={`particle-${i}`} path={path} speed={0.3 + i * 0.1} />
      ))}
    </>
  );
}

export default function NeuralNetwork() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 10], fov: 40 }}
        style={{ background: 'transparent' }}
      >
        <NeuralNetworkScene />
      </Canvas>
    </div>
  );
}
