import React, { useRef, useState, useEffect, useMemo } from 'react'
import * as THREE from 'three';
import { useGLTF, OrbitControls } from '@react-three/drei'
import './App.css'
import { createRoot } from 'react-dom/client'
import { Canvas, useFrame, useLoader, useThree, extend } from '@react-three/fiber'
import { TextureLoader } from 'three/src/loaders/TextureLoader'
import { Shape, ShapeGeometry, BufferGeometry, Float32BufferAttribute, Mesh, MeshBasicMaterial } from 'three';
import { QuickHull } from 'quickhull3d'; // Install this using npm or yarn
import earcut from 'earcut'; // Import earcut for triangulation
import concaveman from 'concaveman';
import { Line } from '@react-three/drei';

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/example.json';

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/0af3b388-5d81-4b78-b7f0-c19786f9f9c1-d245cbfd-82be-4731-8d33-45f7b58be15b.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/0b508d29-c18c-471b-8711-3f114819ea74-c1c5c21c-3271-4ba4-931f-07d25ea25132.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/35061dd6-7af7-4f68-9550-d9a84481c044-20a8d83a-43a7-492a-92c6-8efb945eaed5.json'

//import testData from '/Volumes/apollo11/data/sampled/example6.json'

// import testData from '/Users/mnbucher/Downloads/scene-fig-voxelization.json'
import testData from '/Users/mnbucher/Downloads/scene-fig-teaser.json'

// import testData from '/Users/mnbucher/Downloads/test-doors-windows/0a9c667d-033d-448c-b17c-dc55e6d3c386-2.json'
// import testData from '/Users/mnbucher/Downloads/test-doors-windows/6b774494-78d2-4def-a1df-24e4d907e796-0.json'

// import testData from '/Users/mnbucher/git/stan-24-sgllm/src/frontend/public/example.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2-dedup/7a4521a1-aea5-47c4-9fda-6f8a4dfc0966-3b819f40-7834-48e4-8462-30b7476e12b3.json'

// invalid in atiss pipeline ?
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2-dedup/6f3094b8-689f-4f0c-adb2-748fbc81ec8a-be4506bc-cc92-4ba0-bf41-8dfa30cf4dc1.json'

// invalid meshes ?
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2-dedup/312b2349-10f9-4ff3-bbee-ae179761c21d-7bf0be5c-1923-46cb-8223-19ce3cf10e6a.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2-dedup/a8b833c1-8d0d-4b95-9032-b5fdd2476ecb-127c9c14-f0f8-4d4d-9723-6e7f955584a8.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2-dedup/9a7e6458-3651-43a5-a209-d711efb93772-978ee75d-2e85-4232-9307-21bed4c19678.json'

// import testData from '/Users/mnbucher/Downloads/test.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2/0a17c68b-b74d-4d81-afe4-bc2ed405f0ec-76de6b64-0cab-4c97-ab24-5b55d3811792.json'
//import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-2/4cfd2177-eb18-4006-8d01-435e05c0cbd8-c30109ef-3be7-4038-8607-b459088e2e68.json'

// import testData from '/Volumes/apollo11/data/sampled/example7.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/543516e7-31f4-45b1-9c81-5800d39d2c37-07d12b60-289a-484f-831e-47922f2cb9c5.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/5f501905-6bb6-497e-bd13-59d2478416ad-4b54c2c2-1489-4e1c-bdd0-6847bfb211c3.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/f24ffce5-6f17-40ef-ae61-46a90291b6a9-8e339fbf-14cd-4b12-be32-9d136a3ece7a.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/0af309fd-4a34-4aa2-bbe9-610edadf747e-d66d7594-564a-4b5f-adb6-2cd5758f5348.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/0ab72fdf-ba1b-4df5-8df2-6efcfa7bb53a-6953aea4-1797-4ef5-9efd-ed789149ade0.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/0ba3737a-4fa2-4f70-a7e7-e2629a79483b-bfd0778c-47a7-4105-b93e-916223b4b3b4.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/0c1f82d4-6aa2-401f-ae0d-f053a024aa3b-0fe863aa-2347-4c34-88c1-f71a3ac50df5.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/296aef4e-47fa-447a-b1a6-80d64e5c07d0-7033a638-fc86-4fc4-b891-4fb145ff467b.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/0af309fd-4a34-4aa2-bbe9-610edadf747e-de1883af-ee38-468c-88f5-3bbee10871d8.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/0b508d29-c18c-471b-8711-3f114819ea74-2d1b3401-3bff-473a-b058-2b15c3dab502.json'

// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/103cce55-24d5-4c71-9856-156962e30511-a9c97c39-503a-4dd2-8c40-4c8ea7aeaa6f.json'

// NOT OK: way too big ?
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/00b88e19-d106-4ab8-a322-31c494a0a6b9-360a7b1f-eab3-47f8-8f69-41d405d7c999.json'

// NOT OK: order not correct
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/0c1f82d4-6aa2-401f-ae0d-f053a024aa3b-e46ca246-8d6d-4359-b745-30ae27947455.json'
// import testData from '/Volumes/apollo11/data/3D-FRONT-martin-rooms/0be6d834-32f0-45cc-bbd5-a57924cf5a8a-70194e06-dd23-4c50-9fb9-4f31d2388bce.json'


function fetchData() {
	return testData
}

function App() {
	// const [count, setCount] = useState(0);
	const [showWalls, setShowWalls] = useState(false);
	const [showFloor, setShowFloor] = useState(true);
	const [showBoundingBoxes, setShowBoundingBoxes] = useState(false);
	const [showGrid, setShowGrid] = useState(true);
	const [showAxes, setShowAxes] = useState(true);
	const [showBounds, setShowBounds] = useState(true);
	const [showWindows, setShowWindows] = useState(true);
	const [showObjectOrientations, setShowObjectOrientations] = useState(true);

	const [filterObjectId, setFilterObjectId] = useState("");
	const [data, setData] = useState(null);
	const [selectedAssetID, setSelectedAssetID] = useState(null);
	const [boundingBoxScene, setBoundingBoxScene] = useState(null);
	const [rotationSpeed, setRotationSpeed] = useState(0.01);
	const [isAnimating, setIsAnimating] = useState(false);
	const sceneRef = useRef();
	const [cntObjectsLoaded, setCntObjectsLoaded] = useState(0);
	const [ allObjectsLoaded, setAllObjectsLoaded ] = useState(false);

	useEffect(() => {
		let newData = fetchData();
		setData(newData);
	});

	useEffect(() => {
		if (data && data.objects && data.objects.length > 0) {
			console.log("checking length:", data.objects.length, cntObjectsLoaded);
			if(data.objects.length == cntObjectsLoaded) {
				console.log("all objects loaded!")
				setAllObjectsLoaded(true);
			}
		}
	}, [ cntObjectsLoaded, data ])

  function handleResampleAll() {
		console.log("handleResampleAll")
	}

	function handleResampleObj() {
		console.log("handleResampleObj")
	}

	return (
		<>
			<div className="menu">
				<div className="settings">
					{ testData != null && testData.walls != null ? <button className={showWalls ? "active" : null} type="button" onClick={() => setShowWalls(!showWalls)}>{showWalls ? "Hide Walls" : "Show Walls"}</button> : null }
					<button className={showBoundingBoxes ? "active" : null} type="button" onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}>{showBoundingBoxes ? "Hide Boxes" : "Show Boxes"}</button>
					<button className={showFloor ? "active" : null} type="button" onClick={() => setShowFloor(!showFloor)}>{showFloor ? "Hide Floor" : "Show Floor"}</button>
					<button className={showBounds ? "active" : null} type="button" onClick={() => setShowBounds(!showBounds)}>{showBounds ? "Hide Bounds" : "Show Bounds"}</button>
					<button className={showWindows ? "active" : null} type="button" onClick={() => setShowWindows(!showWindows)}>{showWindows ? "Hide Windows" : "Show Windows"}</button>
					<button className={isAnimating ? "active" : null} type="button" onClick={() => setIsAnimating(!isAnimating)}>{isAnimating ? "Stop Rotation" : "Start Rotation"}</button>
					<button className={showGrid ? "active" : null} type="button" onClick={() => setShowGrid(!showGrid)}>{showGrid ? "Hide Grid" : "Show Grid"}</button>
					<button className={showAxes ? "active" : null} type="button" onClick={() => setShowAxes(!showAxes)}>{showAxes ? "Hide Axes" : "Show Axes"}</button>
					<button className={showObjectOrientations ? "active" : null} type="button" onClick={() => setShowObjectOrientations(!showObjectOrientations)}>{showObjectOrientations ? "Obj. Orient." : "Obj. Orient."}</button>
					<button className="active" onClick={() => handleResampleAll()}>Resample All</button>
				</div>
				<div className="debug-menu">
					<input type="text" onChange={(e) => setFilterObjectId(e.target.value)} value={filterObjectId} placeholder="Filter by ID" />
					{ data && data.objects && <p># of Objects: {data.objects.length}</p> }
					{ selectedAssetID && <p>Selected: {selectedAssetID}</p> }
					{ selectedAssetID && <button className="active" onClick={() => handleResampleObj()}>Resample Obj</button> }
				</div>
			</div>
			<div className="canvas">
				<Canvas ref={ref => { if (ref) sceneRef.current = ref }} >
					<InitialCamera boundingBoxScene={boundingBoxScene} />
					<OrbitControls />
					{ showGrid && <GridHelper /> }
					{ showAxes && <AxesHelper /> }
					<BoundingBoxHelper setBoundingBoxScene={setBoundingBoxScene} allObjectsLoaded={allObjectsLoaded} />
					{ isAnimating ? <SceneAnimator rotationSpeed={rotationSpeed} isAnimating={isAnimating} /> : null }

					{ showFloor && <FloorSlab bounds={testData.bounds_bottom} /> }

					{ showBounds && testData.bounds_bottom != null ? <Bounds bounds={testData.bounds_bottom} /> : null }
					{ showBounds && testData.bounds_top != null ? <Bounds bounds={testData.bounds_top} /> : null }

					{ showWindows && testData.windows != null ? <Windows windows={testData.windows} /> : null }
					
					{ createLighting() }

					{ data && data.objects && createAssetComponents(data, filterObjectId, showBoundingBoxes, showObjectOrientations, selectedAssetID, setSelectedAssetID, setCntObjectsLoaded) }

					{ showWalls && testData.walls != null ? createWallComponents(testData.walls) : null }
					
				</Canvas>
			</div>
		</>
	)
}

function SceneAnimator({ rotationSpeed, isAnimating }) {
  const { scene } = useThree();
  useFrame(() => {
  	console.log("use frame scene animator")
	if (isAnimating) {
	  scene.rotation.y += rotationSpeed;
	}
  });
  return null;
}

function InitialCamera({ boundingBoxScene }) {
  const { camera } = useThree();

  useEffect(() => {
  	console.log("camera UPDATE");
  	camera.position.set(4.0, 4.0, 4.0);
  	camera.lookAt([0, 0, 0]);
		// if (boundingBoxScene) {
		// 	
		// 	const size = new THREE.Vector3();
		// 	boundingBoxScene.getSize(size);

		// 	// Set the camera position relative to the size of the scene
	  //    // const maxSize = Math.max(size.x, size.y, size.z);
		// 	// const distance = maxSize * 2; // Adjust the multiplier as needed for your scene
	  
	  // 	const bboxCenter = new THREE.Vector3();
	  // 	boundingBoxScene.getCenter(bboxCenter)
	  // 	
	  // }
  }, [ camera, boundingBoxScene ]);
}

function createLighting() {
	return (
		<>
			<ambientLight intensity={ 2.5 } />
			<directionalLight position={[0, 2, 0]} intensity={ 2.5 } />
		</>
	);
}

function createAssetComponents(data, filterObjectId, showBoundingBoxes, showObjectOrientations, selectedAssetID, setSelectedAssetID, setCntObjectsLoaded) {
	return (
		<>
			{data.objects.map((obj) => {
				if ((filterObjectId === "" || filterObjectId === undefined) || (filterObjectId !== "" && obj.uid === filterObjectId)) {
					return <AssetComponent
						key={obj.uuid != null ? obj.uuid : obj.instance_id}
						id={obj.uuid != null ? obj.uuid : obj.instance_id}
						file={"./3D-FUTURE-assets/" + (obj.jid != null ? obj.jid : obj.sampled_asset_jid) + "/raw_model.glb"}
						position={obj.pos}
						scale={obj.scale}
						rotation={obj.rot}
						size={obj.size}
						showBoundingBoxes={showBoundingBoxes}
						showObjectOrientations={showObjectOrientations}
						setSelectedAssetID={setSelectedAssetID}
						selectedAssetID={selectedAssetID}
						setCntObjectsLoaded={setCntObjectsLoaded}
					/>
				}
			})}
		</>
	);
}

function createWallComponents(data) {
	return (
		<>
			{data.map((wall) => (
				<WallComponent 
					key={wall.uid} 
					data={wall}
				/>
			))}
		</>
	);
}

function AssetComponent({ id, file, position, scale, rotation, size, showBoundingBoxes, showObjectOrientations, setSelectedAssetID, selectedAssetID, setCntObjectsLoaded}) {
	const { scene } = useGLTF(file);
	const ref = useRef();
	const [ hasLoaded, setHasLoaded ] = useState(false);

	// const quaternion = new THREE.Quaternion(0, 0, 0, 1);
	const quaternion = new THREE.Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]);

	/*useEffect(() => {
		if (ref.current) {
			console.log("rotate! with", hasLoaded ? "true" : "false")
			ref.current.quaternion.copy(quaternion);
		}
	}, [ ref.current, showBoundingBoxes ]);*/

	const handlePointerDown = (event) => {
		setSelectedAssetID(selectedAssetID == id ? null : id);
	};

	useEffect(() => {
		if (!hasLoaded && ref.current) {
			console.log("set has loaded to true")
			setHasLoaded(true);
			setCntObjectsLoaded(prev => prev + 1);
			console.log("showObjectOrientations", showObjectOrientations)
			ref.current.quaternion.copy(quaternion);
		}
	}, [ hasLoaded, ref, setCntObjectsLoaded ]);

	// select object: make some blue color on the texture
	useEffect(() => {
		if (selectedAssetID === id && ref.current) {
			ref.current.traverse((child) => {
				if (child.isMesh) {
					const originalMaterial = child.material;
					child.material = new THREE.MeshStandardMaterial({
						map: originalMaterial.map,
						emissive: new THREE.Color('blue'),
						emissiveIntensity: 0.3,
						transparent: true,
						opacity: 1.0
					});
				}
			});
		} else if (ref.current) {
			ref.current.traverse((child) => {
				if (child.isMesh) {
					const originalMaterial = child.material;
					child.material = new THREE.MeshStandardMaterial({
						map: originalMaterial.map,
						opacity: 1
					});
				}
			});
		}
	}, [ selectedAssetID ]);

	const axesLines = (
		<group quaternion={quaternion}>
			{/* X-axis line (red) */}
			<Line
				points={[[0, 0, 0], [10, 0, 0]]}
				color="red"
				lineWidth={5}
			/>
			{/* Y-axis line (green) */}
			<Line
				points={[[0, 0, 0], [0, 10, 0]]}
				color="green"
				lineWidth={5}
			/>
			{/* Z-axis line (blue) */}
			<Line
				points={[[0, 0, 0], [0, 0, 10]]}
				color="blue"
				lineWidth={5}
			/>
		</group>
	);

	useEffect(() => {
		if (showObjectOrientations && ref.current) {
			const axesHelper = new THREE.AxesHelper(1);
			ref.current.add(axesHelper);
			return () => ref.current.remove(axesHelper);
		}
	}, [showObjectOrientations, ref]);

	let objectToShow;
	if(!showBoundingBoxes) {
		objectToShow = 
			<primitive 
				object={scene.clone(true)}
				onPointerDown={handlePointerDown}
				onLoad={console.log("has loaded")}
			/>;
	} else if (size != null) {
		objectToShow = 
			<mesh 
				position={[ 0, (size[1] / 2), 0 ]}
			>
				<boxGeometry args={size} />
				<meshBasicMaterial color="blue" transparent opacity={0.5} />
			</mesh>;
	}

	return (
		<group
			ref={ref}
			position={[position[0], position[1], position[2]]}
		>
			{objectToShow}
		</group>
	);
}

function parseWallData(data) {
	const geometry = new THREE.BufferGeometry();

	// Parse vertices
	const vertices = new Float32Array(data.xyz);
	geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

	// Parse normals
	const normals = new Float32Array(data.normal);
	geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

	// Parse UVs
	const uvs = new Float32Array(data.uv);
	geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

	// Parse faces (assuming they are triangle indices)
	const indices = new Uint32Array(data.faces);
	geometry.setIndex(new THREE.BufferAttribute(indices, 1));

	geometry.computeBoundingBox();

	return geometry;
}

function WallComponent({ data, opacity }) {
	const ref = useRef();
	const geometry = parseWallData(data);

	return (
		<mesh ref={ref} geometry={geometry}>
			<meshBasicMaterial color="purple" transparent opacity={0.2} />
		</mesh>
	);
}

function GridHelper(props) {
	const ref = useRef();
	return <primitive object={new THREE.GridHelper(100, 100, '#ffb2b2', '#d2d2d2')} ref={ref} {...props} />;
}

function AxesHelper(props) {
	const ref = useRef();
	return <primitive object={new THREE.AxesHelper( 100 )} ref={ref} {...props} />;
	// X = red
	// Y = green
	// Z = blue
}

function BoundingBoxHelper({ setBoundingBoxScene, allObjectsLoaded}) {
	const { scene } = useThree();

	const calculateBoundingBox = () => {
		let box = new THREE.Box3();
		scene.traverse((object) => {
			if (object.isMesh) {

				// weird bug fix to fix floor slab issue
				// const objectPosition = new THREE.Vector3();
				// object.getWorldPosition(objectPosition);

				box.expandByObject(object);

				let boxSize = new THREE.Vector3();
				box.getSize(boxSize);
			}
		});
		setBoundingBoxScene(box);
	};

	useEffect(() => {
		if (allObjectsLoaded) {
			// weird bug fix to fix floor slab issue
			setTimeout(() => {
				console.log("calculating bounding box...")
				calculateBoundingBox();
			}, 100);
		}
	}, [ scene.children, allObjectsLoaded ]);

	return null;
}

const Bounds = ({ bounds }) => {
  return (
	<>
	  {bounds.map(([x, y, z], index) => (
		<mesh key={index} position={[x, y, z]}>
		  {/* Create a small red cube at each 3D point */}
		  <boxGeometry args={[0.1, 0.1, 0.1]} /> {/* Adjust size as needed */}
		  <meshBasicMaterial color={0xff0000} />
		</mesh>
	  ))}
	</>
  );
};

const Windows = ({ windows }) => {
	return (
		<>
		{windows.map((window, windowIndex) => (
			<group key={`window-${windowIndex}`}>
			{window.points.map((point, pointIndex) => (
				<mesh key={`window-${windowIndex}-point-${pointIndex}`} position={[point[0], point[1], point[2]]}>
				<boxGeometry args={[0.1, 0.1, 0.1]} />
				<meshBasicMaterial color={0x00ff00} /> {/* Green color for windows */}
				</mesh>
			))}
			</group>
		))}
		</>
	);
};

function FloorSlab({ bounds }) {
  const texture = useLoader(THREE.TextureLoader, './public/texture.png');
  const [geometry, setGeometry] = useState(null);

  useEffect(() => {
	if (bounds && bounds.length > 0) {

		console.log(bounds);

	  const points2d = bounds.map(point => [point[0], point[2]]);

	  // Step 4: Create the shape
	  const shape = new THREE.Shape();
	  points2d.forEach((pt, index) => {
		if (index === 0) {
		  shape.moveTo(pt[0], pt[1]);
		} else {
		  shape.lineTo(pt[0], pt[1]);
		}
	  });

	  // Step 5: Extrude the shape
	  const extrudeSettings = {
		depth: 0.15, // Thickness of the floor
		bevelEnabled: false,
		steps: 1,
		curveSegments: 12,
	  };
	  const extrudedGeometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);

	  // Rotate the geometry to align with the XZ plane
	  extrudedGeometry.rotateX(Math.PI / 2);

	  // Step 6: Update texture settings
	  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
	  extrudedGeometry.computeBoundingBox();
	  const size = extrudedGeometry.boundingBox.getSize(new THREE.Vector3());
	  texture.repeat.set(size.x / 10, size.z / 10);

	  // Step 7: Update state
	  setGeometry(extrudedGeometry);
	}
  }, [bounds]);

  if (geometry) {
	return (
	  <mesh geometry={geometry}>
		<meshStandardMaterial
		  map={texture}
		  color="#F5DEB3"
		  transparent
		  opacity={0.7}
		  side={THREE.DoubleSide}
		/>
	  </mesh>
	);
  }

  return null;
}

export default App
