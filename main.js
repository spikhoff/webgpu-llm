// app.js

// -------------------------------------------------------------
// Check WebGPU Support
// -------------------------------------------------------------
function checkWebGPUSupport() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser/environment.");
    }
  }
  
  // -------------------------------------------------------------
  // Request GPU Adapter and Device
  // -------------------------------------------------------------
  async function getGPUDevice() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("Failed to get GPU adapter. Check your browser support.");
    }
    const device = await adapter.requestDevice();
    return device;
  }
  
  // -------------------------------------------------------------
  // Create GPU Buffers for data
  // -------------------------------------------------------------
  function createGPUBuffer(device, data, usage) {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage,
    });
    
    // If the usage allows COPY_DST, we can writeBuffer directly.
    if (usage & GPUBufferUsage.COPY_DST) {
      device.queue.writeBuffer(buffer, 0, data);
    }
    
    return buffer;
  }
  
  // -------------------------------------------------------------
  // Create the WGSL Shader Module
  // -------------------------------------------------------------
  function createShaderModule(device) {
    const code = /* wgsl */`
      struct Matrix {
        data: array<f32>,
      };
      
      struct Dimensions {
        rowsA : u32,
        colsA : u32,
        rowsB : u32,
        colsB : u32,
      };
      
      @group(0) @binding(0) var<storage, read> A : Matrix;
      @group(0) @binding(1) var<storage, read> B : Matrix;
      @group(0) @binding(2) var<storage, read_write> C : Matrix;
      @group(0) @binding(3) var<uniform> dims : Dimensions;
  
      // Each invocation handles one (row, col) of the output matrix
      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let row = global_id.y;
        let col = global_id.x;
  
        // Check bounds, in case our workgroup extends beyond the output size
        if (row >= dims.rowsA || col >= dims.colsB) {
          return;
        }
        
        var sum = 0.0;
        for (var i = 0u; i < dims.colsA; i = i + 1u) {
          let aIndex = row * dims.colsA + i;
          let bIndex = i * dims.colsB + col;
          sum = sum + A.data[aIndex] * B.data[bIndex];
        }
        
        let cIndex = row * dims.colsB + col;
        C.data[cIndex] = sum;
      }
    `;
    
    return device.createShaderModule({ code });
  }
  
  // -------------------------------------------------------------
  // Create Compute Pipeline
  // -------------------------------------------------------------
  function createComputePipeline(device, shaderModule) {
    return device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
  }
  
  // -------------------------------------------------------------
  // Create Uniform Buffer (Dimensions)
  // -------------------------------------------------------------
  function createUniformDimensionsBuffer(device, rowsA, colsA, rowsB, colsB) {
    const dimsData = new Uint32Array([rowsA, colsA, rowsB, colsB]);
    const dimsBuffer = device.createBuffer({
      size: dimsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(dimsBuffer, 0, dimsData);
    return dimsBuffer;
  }
  
  // -------------------------------------------------------------
  // Matrix Multiplication (A x B) using a Compute Pass
  // -------------------------------------------------------------
  async function gpuMatrixMultiply(device, bufferA, bufferB, bufferC, dimsBuffer, rowsA, colsB) {
    const shaderModule = createShaderModule(device);
    const computePipeline = createComputePipeline(device, shaderModule);
  
    // Create the BindGroup
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferC } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });
  
    // Encode Commands
    const commandEncoder = device.createCommandEncoder();
  
    // Begin Compute Pass
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
  
    // Workgroup size is 8x8, so calculate dispatch counts
    const workgroupX = Math.ceil(colsB / 8);
    const workgroupY = Math.ceil(rowsA / 8);
    passEncoder.dispatchWorkgroups(workgroupX, workgroupY);
    passEncoder.end();
  
    // Create a staging buffer for reading results
    const bufferSize = bufferC.size;
    const stagingBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    commandEncoder.copyBufferToBuffer(bufferC, 0, stagingBuffer, 0, bufferSize);
  
    // Submit GPU commands
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);
  
    // Wait for GPU to finish
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copyArrayBuffer = stagingBuffer.getMappedRange();
    return new Float32Array(copyArrayBuffer);
  }
  
  // -------------------------------------------------------------
  // Main Execution
  // -------------------------------------------------------------
  async function main() {
    try {
      checkWebGPUSupport();
      const device = await getGPUDevice();
  
      // Example data: A(4x4), B(4x4)
      const A = new Float32Array([
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16
      ]);
      const B = new Float32Array([
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1
      ]);
      const ROWS_A = 4, COLS_A = 4;
      const ROWS_B = 4, COLS_B = 4;
  
      // The output C = (ROWS_A x COLS_B) -> (4 x 4)
      const C = new Float32Array(ROWS_A * COLS_B);
  
      // Create Buffers
      const bufferA = createGPUBuffer(device, A, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const bufferB = createGPUBuffer(device, B, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const bufferC = createGPUBuffer(
        device, 
        C, 
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
  
      // Create Uniform Buffer for dimensions
      const dimsBuffer = createUniformDimensionsBuffer(device, ROWS_A, COLS_A, ROWS_B, COLS_B);
  
      // Perform GPU matrix multiplication
      const result = await gpuMatrixMultiply(
        device,
        bufferA,
        bufferB,
        bufferC,
        dimsBuffer,
        ROWS_A,
        COLS_B
      );
  
      // Log results
      console.log("Result matrix C = A x B:");
      for (let row = 0; row < ROWS_A; row++) {
        const rowVals = [];
        for (let col = 0; col < COLS_B; col++) {
          rowVals.push(result[row * COLS_B + col].toFixed(2));
        }
        console.log(`[ ${rowVals.join(", ")} ]`);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  }
  
  // Run main
  main();
  
