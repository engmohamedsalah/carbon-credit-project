import React, { useRef } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, OrbitControls } from '@react-three/drei';


function Satellite({ position, scale = 1 }) {
  const satRef = useRef();
  const orbitRadius = Math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2);
  const orbitSpeed = 0.2 / orbitRadius; // Slower for farther satellites
  
  useFrame((state) => {
    if (satRef.current) {
      const time = state.clock.elapsedTime * orbitSpeed;
      satRef.current.position.x = Math.cos(time) * orbitRadius;
      satRef.current.position.z = Math.sin(time) * orbitRadius;
      satRef.current.position.y = position[1];
      satRef.current.lookAt(0, 0, 0);
      
      // Rotate solar panels to track sun
      const solarPanelGroup = satRef.current.children.find(child => child.name === 'solarPanels');
      if (solarPanelGroup) {
        solarPanelGroup.rotation.y = Math.sin(time * 0.5) * 0.3;
      }
      
      // Rotate radar dish
      const radarDish = satRef.current.children.find(child => child.name === 'radarDish');
      if (radarDish) {
        radarDish.rotation.z = Math.sin(time * 2) * 0.1;
      }
    }
  });
  
  return (
    <group ref={satRef} scale={[scale * 1.5, scale * 1.5, scale * 1.5]}>
      {/* Main satellite bus - realistic proportions */}
      <group>
        {/* Primary hexagonal structure */}
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
        
        {/* Service module with tanks */}
        <mesh position={[0, -0.04, 0]}>
          <cylinderGeometry args={[0.025, 0.03, 0.02, 12]} />
          <meshStandardMaterial 
            color={'#c0c0c0'} 
            metalness={0.9} 
            roughness={0.1}
          />
        </mesh>
        
        {/* Multi-layer insulation (MLI) - gold foil wrapping */}
        <mesh position={[0, 0.031, 0]}>
          <cylinderGeometry args={[0.041, 0.041, 0.001, 6]} />
          <meshStandardMaterial 
            color={'#FFD700'} 
            metalness={0.7} 
            roughness={0.3}
            emissive={'#FFD700'}
            emissiveIntensity={0.15}
          />
        </mesh>
        {/* Bottom MLI */}
        <mesh position={[0, -0.031, 0]}>
          <cylinderGeometry args={[0.041, 0.041, 0.001, 6]} />
          <meshStandardMaterial 
            color={'#FFA500'} 
            metalness={0.6} 
            roughness={0.4}
            emissive={'#FFA500'}
            emissiveIntensity={0.1}
          />
        </mesh>
      </group>
      
      {/* Main parabolic antenna with realistic structure */}
      <group name="radarDish" position={[0.03, 0.02, 0]} rotation={[0, -Math.PI/6, 0]}>
        {/* Support arm */}
        <mesh>
          <boxGeometry args={[0.015, 0.002, 0.002]} />
          <meshStandardMaterial color={'#505050'} metalness={0.9} />
        </mesh>
        {/* Dish */}
        <mesh position={[0.015, 0, 0]} rotation={[0, 0, Math.PI/2]}>
          <latheGeometry args={[
            [
              [0, 0],
              [0.02, 0.005],
              [0.025, 0.008]
            ].map(p => new THREE.Vector2(...p)),
            32,
            0,
            Math.PI * 2
          ]} />
          <meshStandardMaterial 
            color={'#f0f0f0'} 
            metalness={0.9} 
            roughness={0.05}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* Feed horn and receiver */}
        <mesh position={[0.015, 0, 0]}>
          <sphereGeometry args={[0.003, 8, 8]} />
          <meshStandardMaterial color={'#404040'} metalness={0.95} />
        </mesh>
        {/* Support struts */}
        {[0, Math.PI/2, Math.PI, Math.PI*1.5].map((angle, i) => (
          <mesh key={i} position={[0.015, 0, 0]} rotation={[0, 0, angle]}>
            <boxGeometry args={[0.025, 0.0005, 0.0005]} />
            <meshStandardMaterial color={'#606060'} metalness={0.8} />
          </mesh>
        ))}
      </group>
      
      {/* Omni-directional antenna */}
      <group position={[-0.03, 0.015, 0.02]}>
        <mesh>
          <cylinderGeometry args={[0.001, 0.001, 0.025, 8]} />
          <meshStandardMaterial color={'#606060'} metalness={0.95} />
        </mesh>
        {/* Antenna rings */}
        {[0.005, 0.012, 0.02].map((height, i) => (
          <mesh key={i} position={[0, height, 0]}>
            <torusGeometry args={[0.004, 0.0005, 4, 12]} />
            <meshStandardMaterial color={'#909090'} metalness={0.9} />
          </mesh>
        ))}
      </group>
      
      {/* High-gain antenna with deployment mechanism */}
      <group position={[0, 0.03, 0]}>
        <mesh>
          <cylinderGeometry args={[0.002, 0.002, 0.04, 8]} />
          <meshStandardMaterial 
            color={'#d0d0d0'} 
            metalness={0.95} 
            roughness={0.05}
          />
        </mesh>
        {/* Phased array elements */}
        <mesh position={[0, 0.04, 0]}>
          <boxGeometry args={[0.015, 0.001, 0.015]} />
          <meshStandardMaterial color={'#303030'} metalness={0.8} />
        </mesh>
        {/* Array grid */}
        {[-0.005, 0, 0.005].map((x, i) => (
          [-0.005, 0, 0.005].map((z, j) => (
            <mesh key={`${i}-${j}`} position={[x, 0.041, z]}>
              <cylinderGeometry args={[0.001, 0.001, 0.001]} />
              <meshStandardMaterial color={'#606060'} metalness={0.9} />
            </mesh>
          ))
        )).flat()}
        {/* Beacon */}
        <mesh position={[0, 0.045, 0]}>
          <sphereGeometry args={[0.002, 8, 8]} />
          <meshStandardMaterial 
            color={'#ff0000'} 
            emissive={'#ff0000'}
            emissiveIntensity={0.8}
          />
        </mesh>
      </group>
      
      {/* Large solar panel arrays */}
      <group name="solarPanels">
        {/* Left solar panel array - realistic size */}
        <group position={[-0.1, 0, 0]}>
          {/* Deployment mechanism */}
          <mesh position={[0.05, 0, 0]}>
            <boxGeometry args={[0.01, 0.008, 0.008]} />
            <meshStandardMaterial color={'#404040'} metalness={0.9} />
          </mesh>
          {/* Panel substrate */}
          <mesh castShadow>
            <boxGeometry args={[0.15, 0.001, 0.08]} />
            <meshStandardMaterial 
              color={'#0a0a2e'}
              metalness={0.05}
              roughness={0.95}
            />
          </mesh>
          {/* Frame */}
          <lineLoop>
            <edgesGeometry args={[new THREE.BoxGeometry(0.15, 0.001, 0.08)]} />
            <lineBasicMaterial color={'#808080'} />
          </lineLoop>
          {/* Solar cells - realistic grid */}
          {[...Array(6)].map((_, i) => (
            [...Array(4)].map((_, j) => (
              <mesh key={`l${i}-${j}`} position={[-0.065 + i * 0.022, 0.001, -0.035 + j * 0.018]}>
                <boxGeometry args={[0.02, 0.0005, 0.016]} />
                <meshStandardMaterial 
                  color={'#000033'}
                  metalness={0.4}
                  roughness={0.3}
                  emissive={'#1E90FF'}
                  emissiveIntensity={0.2}
                />
              </mesh>
            ))
          )).flat()}
          {/* Bypass diodes */}
          {[-0.05, 0, 0.05].map((x, i) => (
            <mesh key={`diode${i}`} position={[x, 0.0015, 0.039]}>
              <boxGeometry args={[0.005, 0.0005, 0.002]} />
              <meshStandardMaterial color={'#202020'} />
            </mesh>
          ))}
        </group>
        
        {/* Right solar panel array - realistic size */}
        <group position={[0.1, 0, 0]}>
          {/* Deployment mechanism */}
          <mesh position={[-0.05, 0, 0]}>
            <boxGeometry args={[0.01, 0.008, 0.008]} />
            <meshStandardMaterial color={'#404040'} metalness={0.9} />
          </mesh>
          {/* Panel substrate */}
          <mesh castShadow>
            <boxGeometry args={[0.15, 0.001, 0.08]} />
            <meshStandardMaterial 
              color={'#0a0a2e'}
              metalness={0.05}
              roughness={0.95}
            />
          </mesh>
          {/* Frame */}
          <lineLoop>
            <edgesGeometry args={[new THREE.BoxGeometry(0.15, 0.001, 0.08)]} />
            <lineBasicMaterial color={'#808080'} />
          </lineLoop>
          {/* Solar cells - realistic grid */}
          {[...Array(6)].map((_, i) => (
            [...Array(4)].map((_, j) => (
              <mesh key={`r${i}-${j}`} position={[-0.065 + i * 0.022, 0.001, -0.035 + j * 0.018]}>
                <boxGeometry args={[0.02, 0.0005, 0.016]} />
                <meshStandardMaterial 
                  color={'#000033'}
                  metalness={0.4}
                  roughness={0.3}
                  emissive={'#1E90FF'}
                  emissiveIntensity={0.2}
                />
              </mesh>
            ))
          )).flat()}
          {/* Bypass diodes */}
          {[-0.05, 0, 0.05].map((x, i) => (
            <mesh key={`diode${i}`} position={[x, 0.0015, 0.039]}>
              <boxGeometry args={[0.005, 0.0005, 0.002]} />
              <meshStandardMaterial color={'#202020'} />
            </mesh>
          ))}
        </group>
      </group>
      
      {/* Sensor payload */}
      <group position={[0, -0.02, 0.025]}>
        {/* Camera aperture */}
        <mesh rotation={[Math.PI/2, 0, 0]}>
          <cylinderGeometry args={[0.008, 0.006, 0.01, 12]} />
          <meshStandardMaterial color={'#202020'} metalness={0.95} />
        </mesh>
        {/* Lens */}
        <mesh position={[0, 0, 0.005]}>
          <sphereGeometry args={[0.005, 12, 12, 0, Math.PI]} />
          <meshStandardMaterial 
            color={'#001040'} 
            transparent={true}
            opacity={0.8}
            roughness={0.1}
            metalness={0.5}
          />
        </mesh>
        {/* Sun sensors */}
        {[
          [0.015, 0, 0, 0, 0, Math.PI/2],
          [-0.015, 0, 0, 0, 0, -Math.PI/2],
          [0, 0.015, 0, Math.PI/2, 0, 0],
          [0, -0.015, 0, -Math.PI/2, 0, 0]
        ].map((pos, i) => (
          <mesh key={i} position={pos.slice(0, 3)} rotation={pos.slice(3, 6)}>
            <boxGeometry args={[0.004, 0.004, 0.001]} />
            <meshStandardMaterial color={'#404040'} metalness={0.8} />
          </mesh>
        ))}
      </group>
      
      {/* Battery pack */}
      <mesh position={[0.025, 0, -0.015]}>
        <boxGeometry args={[0.015, 0.02, 0.03]} />
        <meshStandardMaterial 
          color={'#808080'} 
          metalness={0.7} 
          roughness={0.3}
        />
      </mesh>
      
      {/* Propulsion module */}
      <group position={[0, -0.03, -0.035]}>
        {/* Propellant tank */}
        <mesh>
          <sphereGeometry args={[0.015, 12, 12]} />
          <meshStandardMaterial 
            color={'#606060'} 
            metalness={0.8} 
            roughness={0.2}
          />
        </mesh>
        {/* Ion thruster */}
        <mesh position={[0, -0.015, 0]}>
          <cylinderGeometry args={[0.01, 0.012, 0.008, 16]} />
          <meshStandardMaterial 
            color={'#101010'} 
            metalness={0.95} 
            roughness={0.05}
          />
        </mesh>
        {/* Thruster glow */}
        <mesh position={[0, -0.02, 0]}>
          <coneGeometry args={[0.008, 0.01, 8]} />
          <meshBasicMaterial 
            color={'#4169E1'} 
            transparent={true}
            opacity={0.6}
          />
        </mesh>
      </group>
      
      {/* Equipment bay with indicators */}
      <group position={[0, 0, 0.03]}>
        {/* Access panel */}
        <mesh>
          <boxGeometry args={[0.02, 0.015, 0.001]} />
          <meshStandardMaterial color={'#505050'} metalness={0.7} />
        </mesh>
        {/* Status LEDs in realistic arrangement */}
        <group position={[0, 0, 0.001]}>
          <mesh position={[-0.007, 0.005, 0]}>
            <circleGeometry args={[0.001, 8]} />
            <meshBasicMaterial color={'#00ff00'} />
          </mesh>
          <mesh position={[-0.007, 0, 0]}>
            <circleGeometry args={[0.001, 8]} />
            <meshBasicMaterial color={'#00ff00'} />
          </mesh>
          <mesh position={[-0.007, -0.005, 0]}>
            <circleGeometry args={[0.001, 8]} />
            <meshBasicMaterial color={'#ffff00'} />
          </mesh>
          {/* Data port */}
          <mesh position={[0.005, 0, 0]}>
            <boxGeometry args={[0.004, 0.008, 0.001]} />
            <meshStandardMaterial color={'#202020'} />
          </mesh>
        </group>
      </group>
      
      {/* Thermal radiators with heat pipes */}
      <group>
        <mesh position={[0.03, -0.01, 0]} rotation={[0, Math.PI/2, 0]}>
          <boxGeometry args={[0.04, 0.0005, 0.02]} />
          <meshStandardMaterial 
            color={'#c0c0c0'} 
            metalness={0.6} 
            roughness={0.4}
          />
        </mesh>
        {/* Heat pipes */}
        {[-0.01, 0, 0.01].map((z, i) => (
          <mesh key={i} position={[0.03, -0.01, z]} rotation={[0, Math.PI/2, 0]}>
            <cylinderGeometry args={[0.0008, 0.0008, 0.04]} />
            <meshStandardMaterial color={'#808080'} metalness={0.9} />
          </mesh>
        ))}
      </group>
      
      {/* Magnetometer boom */}
      <mesh position={[-0.025, 0.02, 0]} rotation={[0, 0, -Math.PI/6]}>
        <cylinderGeometry args={[0.001, 0.001, 0.03]} />
        <meshStandardMaterial color={'#404040'} metalness={0.8} />
      </mesh>
      <mesh position={[-0.04, 0.03, 0]}>
        <sphereGeometry args={[0.003, 8, 8]} />
        <meshStandardMaterial color={'#606060'} metalness={0.7} />
      </mesh>
    </group>
  );
}

function Earth() {
  const earthRef = useRef();
  
  useFrame((state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.05;
    }
  });
  
  return (
    <group ref={earthRef}>
      {/* Main Earth sphere with texture from URL */}
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
      
      {/* Cloud layer */}
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
    </group>
  );
}

function GlobeScene() {
  return (
    <>
      {/* Much brighter lighting setup for better visibility */}
      <ambientLight intensity={2.0} />
      <directionalLight 
        position={[10, 10, 5]} 
        intensity={3.0} 
        color="#ffffff"
        castShadow 
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
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
      
      {/* Stars background */}
      <Stars 
        radius={300} 
        depth={60} 
        count={5000} 
        factor={4} 
        saturation={0} 
        fade 
        speed={1} 
      />
      
      {/* Earth */}
      <Earth />
      
      {/* Orbiting Satellites - realistic sizes */}
      <Satellite position={[1.8, 0.2, 0]} scale={0.8} />
      <Satellite position={[0, 0.5, 2.2]} scale={0.7} />
      <Satellite position={[-2.5, -0.3, 0.5]} scale={0.9} />
      <Satellite position={[0.5, -0.4, -2.0]} scale={0.75} />
      
      {/* Optional: Mouse controls */}
      <OrbitControls 
        enableZoom={false} 
        enablePan={false} 
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.5}
      />
    </>
  );
}

export default function CarbonGlobe() {
  return (
    <Canvas
      dpr={[1, 2]}
      gl={{ 
        antialias: true, 
        alpha: true, 
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.8
      }}
      camera={{ position: [0, 0, 4], fov: 50 }}
      style={{ width: '100%', height: '100%' }}
      shadows
    >
      <color attach="background" args={['#2a4e7c']} />
      <fog attach="fog" args={['#5a8fc2', 12, 30]} />
      <GlobeScene />
    </Canvas>
  );
}

