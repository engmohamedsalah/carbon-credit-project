import React, { useRef, useMemo, useState, useEffect } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, extend } from '@react-three/fiber';
import { Stars, OrbitControls, shaderMaterial } from '@react-three/drei';

// Aurora shader material for polar lights
const AuroraMaterial = shaderMaterial(
  {
    time: 0,
    color1: new THREE.Color('#00ff88'),
    color2: new THREE.Color('#0088ff'),
    opacity: 0.3
  },
  // Vertex shader
  `
    varying vec2 vUv;
    varying vec3 vPosition;
    void main() {
      vUv = uv;
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  // Fragment shader
  `
    uniform float time;
    uniform vec3 color1;
    uniform vec3 color2;
    uniform float opacity;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      float wave1 = sin(vPosition.x * 10.0 + time * 2.0);
      float wave2 = sin(vPosition.z * 8.0 + time * 1.5);
      float wave3 = cos(vPosition.x * 5.0 - time);
      float noise = (wave1 + wave2 + wave3) / 3.0 * 0.5 + 0.5;
      
      vec3 color = mix(color1, color2, noise);
      float heightFade = 1.0 - abs(vPosition.y * 2.0);
      float alpha = opacity * noise * heightFade * (0.5 + sin(time) * 0.5);
      gl_FragColor = vec4(color, alpha);
    }
  `
);

extend({ AuroraMaterial });

// Floating space dust particles
function SpaceDust() {
  const particlesRef = useRef();
  const particleCount = 500;
  
  const [positions, colors, sizes] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    const color = new THREE.Color();
    
    for (let i = 0; i < particleCount; i++) {
      const radius = 3 + Math.random() * 5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      
      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);
      
      // Varying blue-white colors
      color.setHSL(0.6, 0.3 + Math.random() * 0.3, 0.5 + Math.random() * 0.5);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
      
      sizes[i] = Math.random() * 0.03 + 0.01;
    }
    return [pos, colors, sizes];
  }, []);
  
  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.0002;
      particlesRef.current.rotation.x += 0.0001;
      
      // Pulsating effect
      const time = state.clock.elapsedTime;
      particlesRef.current.material.opacity = 0.3 + Math.sin(time * 0.5) * 0.1;
    }
  });
  
  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particleCount}
          array={colors}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-size"
          count={particleCount}
          array={sizes}
          itemSize={1}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        transparent
        opacity={0.4}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        vertexColors
        depthWrite={false}
      />
    </points>
  );
}

// God rays / light rays effect
function LightRays() {
  const raysRef = useRef();
  const rayGroupRef = useRef();
  
  useFrame((state) => {
    const time = state.clock.elapsedTime;
    
    if (rayGroupRef.current) {
      rayGroupRef.current.rotation.z = Math.sin(time * 0.1) * 0.05;
      rayGroupRef.current.position.x = 4 + Math.sin(time * 0.2) * 0.5;
    }
    
    if (raysRef.current) {
      raysRef.current.children.forEach((ray, i) => {
        ray.material.opacity = 0.02 + Math.sin(time * 0.5 + i) * 0.01;
      });
    }
  });
  
  return (
    <group ref={rayGroupRef} position={[4, 4, -5]} rotation={[0, -Math.PI / 4, 0]}>
      <group ref={raysRef}>
        {[0, 0.5, 1, 1.5, 2].map((offset, i) => (
          <mesh 
            key={i} 
            position={[0, 0, offset * 0.3]} 
            rotation={[0, 0, offset * 0.1]}
          >
            <planeGeometry args={[0.3 + offset * 0.2, 20, 1]} />
            <meshBasicMaterial
              color="#ffffff"
              transparent
              opacity={0.02}
              side={THREE.DoubleSide}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
        ))}
      </group>
    </group>
  );
}

// Your existing Satellite component
function Satellite({ position, scale = 1 }) {
  const satRef = useRef();
  const orbitRadius = Math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2);
  const orbitSpeed = 0.2 / orbitRadius;
  
  useFrame((state) => {
    if (satRef.current) {
      const time = state.clock.elapsedTime * orbitSpeed;
      satRef.current.position.x = Math.cos(time) * orbitRadius;
      satRef.current.position.z = Math.sin(time) * orbitRadius;
      satRef.current.position.y = position[1];
      satRef.current.lookAt(0, 0, 0);
      
      const solarPanelGroup = satRef.current.children.find(child => child.name === 'solarPanels');
      if (solarPanelGroup) {
        solarPanelGroup.rotation.y = Math.sin(time * 0.5) * 0.3;
      }
      
      const radarDish = satRef.current.children.find(child => child.name === 'radarDish');
      if (radarDish) {
        radarDish.rotation.z = Math.sin(time * 2) * 0.1;
      }
    }
  });
  
  return (
    <group ref={satRef} scale={[scale * 1.5, scale * 1.5, scale * 1.5]}>
      {/* Simplified satellite structure for demo */}
      <mesh castShadow receiveShadow>
        <cylinderGeometry args={[0.04, 0.04, 0.06, 6]} />
        <meshStandardMaterial 
          color={'#e8e8e8'} 
          metalness={0.85} 
          roughness={0.15}
          emissive={'#ffffff'}
          emissiveIntensity={0.02}
        />
      </mesh>
      
      <group name="solarPanels">
        <mesh position={[-0.1, 0, 0]} castShadow>
          <boxGeometry args={[0.15, 0.001, 0.08]} />
          <meshStandardMaterial 
            color={'#0a0a2e'}
            metalness={0.05}
            roughness={0.95}
          />
        </mesh>
        <mesh position={[0.1, 0, 0]} castShadow>
          <boxGeometry args={[0.15, 0.001, 0.08]} />
          <meshStandardMaterial 
            color={'#0a0a2e'}
            metalness={0.05}
            roughness={0.95}
          />
        </mesh>
      </group>
    </group>
  );
}

// Enhanced Earth component with aurora
function Earth() {
  const earthRef = useRef();
  const northAuroraRef = useRef();
  const southAuroraRef = useRef();
  
  useFrame((state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.05;
    }
    // Animate aurora
    const time = state.clock.elapsedTime;
    if (northAuroraRef.current) {
      northAuroraRef.current.material.uniforms.time.value = time;
      northAuroraRef.current.rotation.y = time * 0.1;
    }
    if (southAuroraRef.current) {
      southAuroraRef.current.material.uniforms.time.value = time + Math.PI;
      southAuroraRef.current.rotation.y = -time * 0.1;
    }
  });
  
  return (
    <group ref={earthRef}>
      {/* Earth sphere */}
      <mesh>
        <sphereGeometry args={[1, 64, 32]} />
        <meshStandardMaterial 
          map={new THREE.TextureLoader().load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')}
          bumpMap={new THREE.TextureLoader().load('https://unpkg.com/three-globe/example/img/earth-topology.png')}
          bumpScale={0.02}
          roughness={0.8}
          metalness={0.2}
        />
      </mesh>
      
      {/* Clouds */}
      <mesh>
        <sphereGeometry args={[1.01, 32, 16]} />
        <meshStandardMaterial
          map={new THREE.TextureLoader().load('https://unpkg.com/three-globe/example/img/fair_clouds_4k.png')}
          transparent={true}
          opacity={0.6}
          emissive={new THREE.Color('#ffffff')}
          emissiveIntensity={0.3}
        />
      </mesh>
      
      {/* Atmosphere */}
      <mesh scale={[1.1, 1.1, 1.1]}>
        <sphereGeometry args={[1, 60, 30]} />
        <shaderMaterial
          transparent={true}
          side={THREE.BackSide}
          uniforms={{
            color: { value: new THREE.Color('#87CEEB') }
          }}
          vertexShader={`
            varying vec3 vNormal;
            void main() {
              vNormal = normalize(normalMatrix * normal);
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `}
          fragmentShader={`
            varying vec3 vNormal;
            uniform vec3 color;
            void main() {
              float intensity = pow(0.8 - dot(vNormal, vec3(0, 0, 1.0)), 2.5);
              gl_FragColor = vec4(color, intensity * 0.8);
            }
          `}
        />
      </mesh>
      
      {/* Northern Aurora */}
      <mesh ref={northAuroraRef} position={[0, 0.9, 0]}>
        <cylinderGeometry args={[0.3, 0.5, 0.4, 32, 1, true]} />
        <auroraMaterial transparent depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      
      {/* Southern Aurora */}
      <mesh ref={southAuroraRef} position={[0, -0.9, 0]} rotation={[Math.PI, 0, 0]}>
        <cylinderGeometry args={[0.3, 0.5, 0.4, 32, 1, true]} />
        <auroraMaterial 
          transparent 
          depthWrite={false} 
          side={THREE.DoubleSide}
          color1={new THREE.Color('#ff0088')}
          color2={new THREE.Color('#ff00ff')}
          opacity={0.25}
        />
      </mesh>
    </group>
  );
}

function GlobeScene() {
  return (
    <group>
      {/* Enhanced lighting */}
      <ambientLight intensity={2.0} />
      <directionalLight 
        position={[10, 10, 5]} 
        intensity={3.0} 
        color="#ffffff"
        castShadow 
        shadow-mapSize={[2048, 2048]}
      />
      <pointLight position={[-10, -10, -5]} intensity={1.5} color="#ffffff" />
      <pointLight position={[5, 5, 5]} intensity={1.2} color="#87CEEB" />
      <spotLight
        position={[0, 5, 0]}
        angle={0.6}
        penumbra={0.5}
        intensity={2.0}
        color="#ffffff"
        castShadow
      />
      
      {/* Visual effects */}
      <LightRays />
      
      {/* Enhanced star fields */}
      <Stars 
        radius={300} 
        depth={60} 
        count={10000} 
        factor={7} 
        saturation={0} 
        fade 
        speed={0.3} 
      />
      
      <Stars 
        radius={200} 
        depth={100} 
        count={500} 
        factor={3} 
        saturation={0.3} 
        fade 
        speed={2}
        color="#4a7ba7"
      />
      
      <SpaceDust />
      
      <Earth />
      
      {/* Satellites */}
      <Satellite position={[1.8, 0.2, 0]} scale={0.8} />
      <Satellite position={[0, 0.5, 2.2]} scale={0.7} />
      <Satellite position={[-2.5, -0.3, 0.5]} scale={0.9} />
      <Satellite position={[0.5, -0.4, -2.0]} scale={0.75} />
      
      <OrbitControls 
        enableZoom={false} 
        enablePan={false} 
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.5}
      />
    </group>
  );
}

export default function CarbonGlobeEnhanced() {
  const [bgGradient, setBgGradient] = useState('radial-gradient(ellipse at center, #1a3a5c 0%, #0d1e3a 40%, #030a1a 100%)');
  
  useEffect(() => {
    // Subtle gradient animation
    const interval = setInterval(() => {
      const time = Date.now() * 0.00005;
      const hue1 = 210 + Math.sin(time) * 15;
      const hue2 = 220 + Math.cos(time) * 10;
      const centerLight = 22 + Math.sin(time * 2) * 3;
      const midLight = 15 + Math.cos(time * 1.5) * 3;
      
      setBgGradient(
        `radial-gradient(ellipse at center, 
          hsl(${hue1}, 45%, ${centerLight}%) 0%, 
          hsl(${hue2}, 50%, ${midLight}%) 40%, 
          #030a1a 100%)`
      );
    }, 50);
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Animated gradient background */}
      <div 
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          background: bgGradient,
          transition: 'background 2s ease',
          zIndex: 0
        }} 
      />
      <Canvas
        dpr={[1, 2]}
        gl={{ 
          antialias: true, 
          alpha: true, 
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.8,
          powerPreference: "high-performance"
        }}
        camera={{ position: [0, 0, 4], fov: 50 }}
        style={{ width: '100%', height: '100%', position: 'relative', zIndex: 1 }}
        shadows
      >
        <GlobeScene />
      </Canvas>
    </div>
  );
}
