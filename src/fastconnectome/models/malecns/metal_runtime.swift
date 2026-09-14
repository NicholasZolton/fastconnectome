import Foundation
import Metal

private let shaderSource = #"""
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint slots;
    long clock;
    float dt;
    int delay;
    int refractoryTicks;
    float adaptationJump;
    float adaptationTau;
    uint eventCapacity;
};

inline void evolve(
    uint i,
    long now,
    float current,
    device float* v,
    device float* g,
    device short* refractory,
    device long* last,
    device const float* rest,
    device float* adaptation,
    device const float* voltageDecay,
    device const float* conductanceDecay,
    device const float* adaptationDecay,
    constant Params& params
) {
    long elapsed = now - last[i];
    if (elapsed <= 0) return;
    int frozen = refractory[i] > 0 ? int(refractory[i]) - 1 : 0;
    int skipped = int(min(elapsed, long(frozen)));
    if (skipped > 0 && adaptation[i] > 0.0f) {
        adaptation[i] *= skipped < 1024
            ? adaptationDecay[skipped]
            : exp(-params.dt * float(skipped) / params.adaptationTau);
    }
    refractory[i] = elapsed >= long(refractory[i])
        ? short(0)
        : short(long(refractory[i]) - elapsed);
    elapsed -= long(skipped);
    if (elapsed > 0) {
        float a = elapsed < 1024
            ? voltageDecay[elapsed]
            : exp(-params.dt * float(elapsed) / 20.0f);
        float b = elapsed < 1024
            ? conductanceDecay[elapsed]
            : exp(-params.dt * float(elapsed) / 5.0f);
        v[i] = rest[i] + (v[i] - rest[i]) * a
            + current * (1.0f - a) + g[i] * (a - b) / 3.0f;
        g[i] *= b;
        if (adaptation[i] > 0.0f) {
            float c = elapsed < 1024
                ? adaptationDecay[elapsed]
                : exp(-params.dt * float(elapsed) / params.adaptationTau);
            v[i] -= adaptation[i] * params.adaptationTau
                / (params.adaptationTau - 20.0f) * (c - a);
            adaptation[i] *= c;
        }
    }
    last[i] = now;
}

inline void evolveOne(
    uint i,
    long now,
    float current,
    device float* v,
    device float* g,
    device short* refractory,
    device long* last,
    device const float* rest,
    device float* adaptation,
    device const float* voltageDecay,
    device const float* conductanceDecay,
    device const float* adaptationDecay
) {
    if (refractory[i] > 1) {
        if (adaptation[i] > 0.0f) adaptation[i] *= adaptationDecay[1];
        refractory[i] -= 1;
        last[i] = now;
        return;
    }
    if (refractory[i] == 1) refractory[i] = 0;
    float a = voltageDecay[1];
    float b = conductanceDecay[1];
    v[i] = rest[i] + (v[i] - rest[i]) * a
        + current * (1.0f - a) + g[i] * (a - b) / 3.0f;
    g[i] *= b;
    if (adaptation[i] > 0.0f) {
        float c = adaptationDecay[1];
        v[i] -= adaptation[i] * 200.0f / 180.0f * (c - a);
        adaptation[i] *= c;
    }
    last[i] = now;
}

kernel void applyDrive(
    device float* v [[buffer(3)]],
    device float* g [[buffer(4)]],
    device short* refractory [[buffer(5)]],
    device const float* drive [[buffer(6)]],
    device float* previous [[buffer(7)]],
    device uchar* flags [[buffer(15)]],
    device long* last [[buffer(16)]],
    device const float* rest [[buffer(19)]],
    device float* adaptation [[buffer(20)]],
    constant Params& params [[buffer(25)]],
    device const float* voltageDecay [[buffer(27)]],
    device const float* conductanceDecay [[buffer(28)]],
    device const float* adaptationDecay [[buffer(29)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n || drive[gid] == previous[gid]) return;
    evolve(
        gid, params.clock - 1, previous[gid], v, g, refractory, last,
        rest, adaptation, voltageDecay, conductanceDecay, adaptationDecay, params
    );
    previous[gid] = drive[gid];
    flags[gid] = 1;
}

kernel void processActive(
    device float* v [[buffer(3)]],
    device float* g [[buffer(4)]],
    device short* refractory [[buffer(5)]],
    device const float* drive [[buffer(6)]],
    device int* spikeQueue [[buffer(8)]],
    device atomic_uint* queueArguments [[buffer(9)]],
    device int* counts [[buffer(10)]],
    device atomic_uint* eventCount [[buffer(14)]],
    device uchar* flags [[buffer(15)]],
    device long* last [[buffer(16)]],
    device const uchar* kcMask [[buffer(17)]],
    device const float* rest [[buffer(19)]],
    device float* adaptation [[buffer(20)]],
    device int* eventIndices [[buffer(22)]],
    device long* eventClocks [[buffer(23)]],
    constant Params& params [[buffer(25)]],
    device uchar* spiked [[buffer(26)]],
    device const float* voltageDecay [[buffer(27)]],
    device const float* conductanceDecay [[buffer(28)]],
    device const float* adaptationDecay [[buffer(29)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n || flags[gid] == 0) return;
    uint i = gid;
    if (params.clock == 0) {
        evolve(
            i, params.clock, drive[i], v, g, refractory, last, rest,
            adaptation, voltageDecay, conductanceDecay, adaptationDecay, params
        );
    } else {
        evolveOne(
            i, params.clock, drive[i], v, g, refractory, last, rest,
            adaptation, voltageDecay, conductanceDecay, adaptationDecay
        );
    }
    uint future = uint((params.clock + params.delay) % params.slots);
    if (refractory[i] == 0 && v[i] > -45.0f) {
        uint queueIndex = atomic_fetch_add_explicit(
            &queueArguments[future * 3], 1u, memory_order_relaxed
        );
        spikeQueue[future * params.n + queueIndex] = int(i);
        counts[i] += 1;
        spiked[i] = 1;
        if (kcMask[i]) {
            adaptation[i] += params.adaptationJump;
            uint eventIndex = atomic_fetch_add_explicit(
                eventCount, 1u, memory_order_relaxed
            );
            if (eventIndex < params.eventCapacity) {
                eventIndices[eventIndex] = int(i);
                eventClocks[eventIndex] = params.clock;
            }
        }
    }
    float gap = -45.0f - rest[i];
    bool canFire = v[i] > -45.0f || drive[i] > gap || drive[i] + g[i] > gap;
    if (!canFire) flags[i] = 0;
}

kernel void scatterEvents(
    device const long* ptr [[buffer(0)]],
    device const int* post [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const int* spikeQueue [[buffer(8)]],
    device const uchar* modulationMask [[buffer(18)]],
    device atomic_float* incoming [[buffer(21)]],
    device atomic_float* modulationIncoming [[buffer(24)]],
    constant Params& params [[buffer(25)]],
    uint queueIndex [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]
) {
    uint slot = uint(params.clock % params.slots);
    uint i = uint(spikeQueue[slot * params.n + queueIndex]);
    bool modulator = modulationMask[i] != 0;
    for (long edge = ptr[i] + long(lane); edge < ptr[i + 1]; edge += long(width)) {
        uint j = uint(post[edge]);
        if (modulator) {
            atomic_fetch_add_explicit(
                &modulationIncoming[j], abs(weight[edge]) / 0.275f,
                memory_order_relaxed
            );
        } else {
            atomic_fetch_add_explicit(&incoming[j], weight[edge], memory_order_relaxed);
        }
    }
}

kernel void processTouched(
    device float* v [[buffer(3)]],
    device float* g [[buffer(4)]],
    device short* refractory [[buffer(5)]],
    device const float* drive [[buffer(6)]],
    device atomic_uint* queueArguments [[buffer(9)]],
    device float* modulation [[buffer(11)]],
    device long* modulationLast [[buffer(12)]],
    device atomic_float* modulationIncoming [[buffer(24)]],
    device uchar* flags [[buffer(15)]],
    device long* last [[buffer(16)]],
    device const float* rest [[buffer(19)]],
    device float* adaptation [[buffer(20)]],
    device atomic_float* incoming [[buffer(21)]],
    constant Params& params [[buffer(25)]],
    device uchar* spiked [[buffer(26)]],
    device const float* voltageDecay [[buffer(27)]],
    device const float* conductanceDecay [[buffer(28)]],
    device const float* adaptationDecay [[buffer(29)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        atomic_store_explicit(
            &queueArguments[uint(params.clock % params.slots) * 3],
            0u,
            memory_order_relaxed
        );
    }
    if (gid >= params.n) return;
    uint i = gid;
    float observed = atomic_load_explicit(&incoming[i], memory_order_relaxed);
    float amount = observed == 0.0f
        ? 0.0f
        : atomic_exchange_explicit(&incoming[i], 0.0f, memory_order_relaxed);
    if (amount != 0.0f) {
        evolve(
            i, params.clock, drive[i], v, g, refractory, last, rest,
            adaptation, voltageDecay, conductanceDecay, adaptationDecay, params
        );
        if (refractory[i] == 0) {
            g[i] += amount;
            flags[i] = 1;
        }
    }
    float observedModulation = atomic_load_explicit(
        &modulationIncoming[i], memory_order_relaxed
    );
    float modulationAmount = observedModulation == 0.0f
        ? 0.0f
        : atomic_exchange_explicit(
            &modulationIncoming[i], 0.0f, memory_order_relaxed
        );
    if (modulationAmount != 0.0f) {
        long elapsed = params.clock - modulationLast[i];
        modulation[i] *= exp(-params.dt * float(elapsed) / 100.0f);
        modulation[i] += modulationAmount;
        modulationLast[i] = params.clock;
    }
    if (spiked[i]) {
        v[i] = rest[i];
        g[i] = 0.0f;
        refractory[i] = short(params.refractoryTicks);
        spiked[i] = 0;
    }
}

kernel void materialize(
    device float* v [[buffer(3)]],
    device float* g [[buffer(4)]],
    device short* refractory [[buffer(5)]],
    device const float* drive [[buffer(6)]],
    device long* last [[buffer(16)]],
    device const float* rest [[buffer(19)]],
    device float* adaptation [[buffer(20)]],
    constant Params& params [[buffer(25)]],
    device const float* voltageDecay [[buffer(27)]],
    device const float* conductanceDecay [[buffer(28)]],
    device const float* adaptationDecay [[buffer(29)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.n) {
        evolve(
            gid, params.clock - 1, drive[gid], v, g, refractory, last,
            rest, adaptation, voltageDecay, conductanceDecay, adaptationDecay, params
        );
    }
}
"""#

private struct Params {
    var n: UInt32
    var slots: UInt32
    var clock: Int64
    var dt: Float
    var delay: Int32
    var refractoryTicks: Int32
    var adaptationJump: Float
    var adaptationTau: Float
    var eventCapacity: UInt32
}

private final class Runtime {
    let n: Int
    let edgeCount: Int
    let slots: Int
    let eventCapacity: Int
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelines: [String: MTLComputePipelineState]
    var clock: Int64

    let ptr: MTLBuffer
    let post: MTLBuffer
    let weight: MTLBuffer
    let v: MTLBuffer
    let g: MTLBuffer
    let refractory: MTLBuffer
    let drive: MTLBuffer
    let previousDrive: MTLBuffer
    let spikeQueue: MTLBuffer
    let queueArguments: MTLBuffer
    let counts: MTLBuffer
    let modulation: MTLBuffer
    let modulationLast: MTLBuffer
    let flags: MTLBuffer
    let last: MTLBuffer
    let kcMask: MTLBuffer
    let modulationMask: MTLBuffer
    let rest: MTLBuffer
    let adaptation: MTLBuffer
    let incoming: MTLBuffer
    let eventIndices: MTLBuffer
    let eventClocks: MTLBuffer
    let modulationIncoming: MTLBuffer
    let eventCount: MTLBuffer
    let spiked: MTLBuffer
    let voltageDecay: MTLBuffer
    let conductanceDecay: MTLBuffer
    let adaptationDecay: MTLBuffer

    init(
        n: Int,
        edgeCount: Int,
        slots: Int,
        clock: Int64,
        ptr sourcePtr: UnsafePointer<Int64>,
        post sourcePost: UnsafePointer<Int32>,
        weight sourceWeight: UnsafePointer<Float>,
        v sourceV: UnsafePointer<Float>,
        g sourceG: UnsafePointer<Float>,
        refractory sourceRefractory: UnsafePointer<Int16>,
        previousDrive sourcePreviousDrive: UnsafePointer<Float>,
        spikeQueue sourceSpikeQueue: UnsafePointer<Int32>,
        queueCount sourceQueueCount: UnsafePointer<Int32>,
        flags sourceFlags: UnsafePointer<UInt8>,
        last sourceLast: UnsafePointer<Int64>,
        kcMask sourceKcMask: UnsafePointer<UInt8>,
        modulationMask sourceModulationMask: UnsafePointer<UInt8>,
        rest sourceRest: UnsafePointer<Float>,
        adaptation sourceAdaptation: UnsafePointer<Float>,
        modulation sourceModulation: UnsafePointer<Float>,
        modulationLast sourceModulationLast: UnsafePointer<Int64>
    ) throws {
        guard let selectedDevice = MTLCreateSystemDefaultDevice(),
              let selectedQueue = selectedDevice.makeCommandQueue() else {
            throw RuntimeError.message("Metal is unavailable")
        }
        self.n = n
        self.edgeCount = edgeCount
        self.slots = slots
        self.eventCapacity = n * 5
        self.device = selectedDevice
        self.commandQueue = selectedQueue
        self.clock = clock

        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        let library = try selectedDevice.makeLibrary(source: shaderSource, options: options)
        var builtPipelines: [String: MTLComputePipelineState] = [:]
        for name in ["applyDrive", "processActive", "scatterEvents", "processTouched", "materialize"] {
            guard let function = library.makeFunction(name: name) else {
                throw RuntimeError.message("Missing Metal function \(name)")
            }
            builtPipelines[name] = try selectedDevice.makeComputePipelineState(function: function)
        }
        self.pipelines = builtPipelines

        self.ptr = try Runtime.privateBuffer(
            selectedDevice, selectedQueue, sourcePtr, count: n + 1
        )
        self.post = try Runtime.privateBuffer(
            selectedDevice, selectedQueue, sourcePost, count: edgeCount
        )
        self.weight = Runtime.sharedBuffer(selectedDevice, sourceWeight, count: edgeCount)
        self.v = Runtime.sharedBuffer(selectedDevice, sourceV, count: n)
        self.g = Runtime.sharedBuffer(selectedDevice, sourceG, count: n)
        self.refractory = Runtime.sharedBuffer(selectedDevice, sourceRefractory, count: n)
        self.drive = Runtime.zeroBuffer(selectedDevice, bytes: n * MemoryLayout<Float>.stride)
        self.previousDrive = Runtime.sharedBuffer(selectedDevice, sourcePreviousDrive, count: n)
        self.spikeQueue = Runtime.sharedBuffer(
            selectedDevice, sourceSpikeQueue, count: n * slots
        )
        var arguments = [UInt32](repeating: 1, count: slots * 3)
        for slot in 0..<slots {
            arguments[slot * 3] = UInt32(sourceQueueCount[slot])
        }
        self.queueArguments = arguments.withUnsafeBufferPointer {
            Runtime.sharedBuffer(selectedDevice, $0.baseAddress!, count: arguments.count)
        }
        self.counts = Runtime.zeroBuffer(selectedDevice, bytes: n * MemoryLayout<Int32>.stride)
        self.modulation = Runtime.sharedBuffer(selectedDevice, sourceModulation, count: n)
        self.modulationLast = Runtime.sharedBuffer(
            selectedDevice, sourceModulationLast, count: n
        )
        self.flags = Runtime.sharedBuffer(selectedDevice, sourceFlags, count: n)
        self.last = Runtime.sharedBuffer(selectedDevice, sourceLast, count: n)
        self.kcMask = try Runtime.privateBuffer(
            selectedDevice, selectedQueue, sourceKcMask, count: n
        )
        self.modulationMask = try Runtime.privateBuffer(
            selectedDevice, selectedQueue, sourceModulationMask, count: n
        )
        self.rest = try Runtime.privateBuffer(
            selectedDevice, selectedQueue, sourceRest, count: n
        )
        self.adaptation = Runtime.sharedBuffer(selectedDevice, sourceAdaptation, count: n)
        self.incoming = Runtime.zeroBuffer(selectedDevice, bytes: n * MemoryLayout<Float>.stride)
        self.eventIndices = Runtime.zeroBuffer(
            selectedDevice, bytes: eventCapacity * MemoryLayout<Int32>.stride
        )
        self.eventClocks = Runtime.zeroBuffer(
            selectedDevice, bytes: eventCapacity * MemoryLayout<Int64>.stride
        )
        self.modulationIncoming = Runtime.zeroBuffer(
            selectedDevice, bytes: n * MemoryLayout<Float>.stride
        )
        self.eventCount = Runtime.zeroBuffer(selectedDevice, bytes: MemoryLayout<UInt32>.stride)
        self.spiked = Runtime.zeroBuffer(selectedDevice, bytes: n)

        var voltage = [Float](repeating: 0, count: 1024)
        var conductance = [Float](repeating: 0, count: 1024)
        var adaptationValues = [Float](repeating: 0, count: 1024)
        for index in 0..<1024 {
            voltage[index] = expf(-0.1 * Float(index) / 20.0)
            conductance[index] = expf(-0.1 * Float(index) / 5.0)
            adaptationValues[index] = expf(-0.1 * Float(index) / 200.0)
        }
        self.voltageDecay = try voltage.withUnsafeBufferPointer {
            try Runtime.privateBuffer(selectedDevice, selectedQueue, $0.baseAddress!, count: 1024)
        }
        self.conductanceDecay = try conductance.withUnsafeBufferPointer {
            try Runtime.privateBuffer(selectedDevice, selectedQueue, $0.baseAddress!, count: 1024)
        }
        self.adaptationDecay = try adaptationValues.withUnsafeBufferPointer {
            try Runtime.privateBuffer(selectedDevice, selectedQueue, $0.baseAddress!, count: 1024)
        }
    }

    func advance(
        drive sourceDrive: UnsafePointer<Float>,
        ticks: Int,
        outputCounts: UnsafeMutablePointer<Int32>,
        outputEventIndices: UnsafeMutablePointer<Int32>,
        outputEventClocks: UnsafeMutablePointer<Int64>,
        outputEventCount: UnsafeMutablePointer<Int32>
    ) throws -> Double {
        guard ticks > 0 && ticks <= 100 else {
            throw RuntimeError.message("Metal advances must contain 1...100 ticks")
        }
        memcpy(drive.contents(), sourceDrive, n * MemoryLayout<Float>.stride)
        memset(counts.contents(), 0, n * MemoryLayout<Int32>.stride)
        memset(eventCount.contents(), 0, MemoryLayout<UInt32>.stride)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RuntimeError.message("Could not create Metal command encoder")
        }
        bind(encoder)
        var params = Params(
            n: UInt32(n), slots: UInt32(slots), clock: clock, dt: 0.1,
            delay: 18, refractoryTicks: 22, adaptationJump: 8,
            adaptationTau: 200, eventCapacity: UInt32(eventCapacity)
        )
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 25)
        dispatch(encoder, pipeline: pipeline("applyDrive"), threads: n)
        encoder.memoryBarrier(scope: .buffers)
        for _ in 0..<ticks {
            params.clock = clock
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 25)
            dispatch(encoder, pipeline: pipeline("processActive"), threads: n)
            encoder.setComputePipelineState(pipeline("scatterEvents"))
            encoder.dispatchThreadgroups(
                indirectBuffer: queueArguments,
                indirectBufferOffset: Int(clock % Int64(slots)) * 12,
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )
            encoder.memoryBarrier(scope: .buffers)
            dispatch(encoder, pipeline: pipeline("processTouched"), threads: n)
            encoder.memoryBarrier(scope: .buffers)
            clock += 1
        }
        params.clock = clock
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 25)
        dispatch(encoder, pipeline: pipeline("materialize"), threads: n)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        memcpy(outputCounts, counts.contents(), n * MemoryLayout<Int32>.stride)
        let capturedEvents = Int(eventCount.contents().assumingMemoryBound(to: UInt32.self).pointee)
        guard capturedEvents <= eventCapacity else {
            throw RuntimeError.message("KC spike event capacity exceeded")
        }
        memcpy(
            outputEventIndices,
            eventIndices.contents(),
            capturedEvents * MemoryLayout<Int32>.stride
        )
        memcpy(
            outputEventClocks,
            eventClocks.contents(),
            capturedEvents * MemoryLayout<Int64>.stride
        )
        outputEventCount.pointee = Int32(capturedEvents)
        return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }

    func bind(_ encoder: MTLComputeCommandEncoder) {
        let buffers: [(Int, MTLBuffer)] = [
            (0, ptr), (1, post), (2, weight), (3, v), (4, g),
            (5, refractory), (6, drive), (7, previousDrive), (8, spikeQueue),
            (9, queueArguments), (10, counts), (11, modulation),
            (12, modulationLast), (14, eventCount), (15, flags), (16, last),
            (17, kcMask), (18, modulationMask), (19, rest), (20, adaptation),
            (21, incoming), (22, eventIndices), (23, eventClocks),
            (24, modulationIncoming), (26, spiked), (27, voltageDecay),
            (28, conductanceDecay), (29, adaptationDecay),
        ]
        for (index, buffer) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
    }

    func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        threads: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        let width = min(1024, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        pipelines[name]!
    }

    func readState(
        v outputV: UnsafeMutablePointer<Float>,
        g outputG: UnsafeMutablePointer<Float>,
        refractory outputRefractory: UnsafeMutablePointer<Int16>,
        previousDrive outputPreviousDrive: UnsafeMutablePointer<Float>,
        spikeQueue outputQueue: UnsafeMutablePointer<Int32>,
        queueCount outputQueueCount: UnsafeMutablePointer<Int32>,
        flags outputFlags: UnsafeMutablePointer<UInt8>,
        last outputLast: UnsafeMutablePointer<Int64>,
        adaptation outputAdaptation: UnsafeMutablePointer<Float>,
        modulation outputModulation: UnsafeMutablePointer<Float>,
        modulationLast outputModulationLast: UnsafeMutablePointer<Int64>
    ) {
        Runtime.copy(v, to: outputV, count: n)
        Runtime.copy(g, to: outputG, count: n)
        Runtime.copy(refractory, to: outputRefractory, count: n)
        Runtime.copy(previousDrive, to: outputPreviousDrive, count: n)
        Runtime.copy(spikeQueue, to: outputQueue, count: n * slots)
        let arguments = queueArguments.contents().assumingMemoryBound(to: UInt32.self)
        for slot in 0..<slots {
            outputQueueCount[slot] = Int32(arguments[slot * 3])
        }
        Runtime.copy(flags, to: outputFlags, count: n)
        Runtime.copy(last, to: outputLast, count: n)
        Runtime.copy(adaptation, to: outputAdaptation, count: n)
        Runtime.copy(modulation, to: outputModulation, count: n)
        Runtime.copy(modulationLast, to: outputModulationLast, count: n)
    }

    func writeState(
        clock: Int64,
        v sourceV: UnsafePointer<Float>,
        g sourceG: UnsafePointer<Float>,
        refractory sourceRefractory: UnsafePointer<Int16>,
        previousDrive sourcePreviousDrive: UnsafePointer<Float>,
        spikeQueue sourceQueue: UnsafePointer<Int32>,
        queueCount sourceQueueCount: UnsafePointer<Int32>,
        flags sourceFlags: UnsafePointer<UInt8>,
        last sourceLast: UnsafePointer<Int64>,
        adaptation sourceAdaptation: UnsafePointer<Float>,
        modulation sourceModulation: UnsafePointer<Float>,
        modulationLast sourceModulationLast: UnsafePointer<Int64>
    ) {
        self.clock = clock
        Runtime.copy(sourceV, to: v, count: n)
        Runtime.copy(sourceG, to: g, count: n)
        Runtime.copy(sourceRefractory, to: refractory, count: n)
        Runtime.copy(sourcePreviousDrive, to: previousDrive, count: n)
        Runtime.copy(sourceQueue, to: spikeQueue, count: n * slots)
        let arguments = queueArguments.contents().assumingMemoryBound(to: UInt32.self)
        for slot in 0..<slots {
            arguments[slot * 3] = UInt32(sourceQueueCount[slot])
            arguments[slot * 3 + 1] = 1
            arguments[slot * 3 + 2] = 1
        }
        Runtime.copy(sourceFlags, to: flags, count: n)
        Runtime.copy(sourceLast, to: last, count: n)
        Runtime.copy(sourceAdaptation, to: adaptation, count: n)
        Runtime.copy(sourceModulation, to: modulation, count: n)
        Runtime.copy(sourceModulationLast, to: modulationLast, count: n)
        memset(incoming.contents(), 0, incoming.length)
        memset(modulationIncoming.contents(), 0, modulationIncoming.length)
        memset(spiked.contents(), 0, spiked.length)
    }

    func writeWeights(_ source: UnsafePointer<Float>, count: Int) throws {
        guard count == edgeCount else {
            throw RuntimeError.message("Weight array length mismatch")
        }
        Runtime.copy(source, to: weight, count: edgeCount)
    }

    func updateWeights(
        edges: UnsafePointer<Int64>,
        values: UnsafePointer<Float>,
        count: Int
    ) throws {
        let destination = weight.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<count {
            let edge = Int(edges[index])
            guard edge >= 0 && edge < edgeCount else {
                throw RuntimeError.message("Plastic edge index out of range")
            }
            destination[edge] = values[index]
        }
    }

    private static func sharedBuffer<Element>(
        _ device: MTLDevice,
        _ source: UnsafePointer<Element>,
        count: Int
    ) -> MTLBuffer {
        device.makeBuffer(
            bytes: source,
            length: count * MemoryLayout<Element>.stride,
            options: .storageModeShared
        )!
    }

    private static func privateBuffer<Element>(
        _ device: MTLDevice,
        _ queue: MTLCommandQueue,
        _ source: UnsafePointer<Element>,
        count: Int
    ) throws -> MTLBuffer {
        let bytes = count * MemoryLayout<Element>.stride
        let staging = device.makeBuffer(
            bytes: source, length: bytes, options: .storageModeShared
        )!
        let result = device.makeBuffer(length: bytes, options: .storageModePrivate)!
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw RuntimeError.message("Could not stage private Metal buffer")
        }
        encoder.copy(
            from: staging, sourceOffset: 0, to: result, destinationOffset: 0, size: bytes
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        return result
    }

    private static func zeroBuffer(_ device: MTLDevice, bytes: Int) -> MTLBuffer {
        let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buffer.contents(), 0, bytes)
        return buffer
    }

    private static func copy<Element>(
        _ buffer: MTLBuffer,
        to destination: UnsafeMutablePointer<Element>,
        count: Int
    ) {
        memcpy(destination, buffer.contents(), count * MemoryLayout<Element>.stride)
    }

    private static func copy<Element>(
        _ source: UnsafePointer<Element>,
        to buffer: MTLBuffer,
        count: Int
    ) {
        memcpy(buffer.contents(), source, count * MemoryLayout<Element>.stride)
    }
}

private enum RuntimeError: Error {
    case message(String)
}

private var nextHandle: UInt64 = 1
private var runtimes: [UInt64: Runtime] = [:]
private let runtimeLock = NSLock()
private var lastError: NSString = ""

private func recordError(_ error: Error) {
    lastError = String(describing: error) as NSString
}

private func withRuntime<Result>(
    _ handle: UInt64,
    _ operation: (Runtime) throws -> Result
) throws -> Result {
    runtimeLock.lock()
    let runtime = runtimes[handle]
    runtimeLock.unlock()
    guard let runtime else {
        throw RuntimeError.message("Unknown Metal runtime handle")
    }
    return try operation(runtime)
}

@_cdecl("fc_metal_available")
public func fcMetalAvailable() -> Int32 {
    MTLCreateSystemDefaultDevice() == nil ? 0 : 1
}

@_cdecl("fc_last_error")
public func fcLastError() -> UnsafePointer<CChar>? {
    lastError.utf8String
}

@_cdecl("fc_create")
public func fcCreate(
    _ n: Int32,
    _ edgeCount: Int64,
    _ slots: Int32,
    _ clock: Int64,
    _ ptr: UnsafePointer<Int64>,
    _ post: UnsafePointer<Int32>,
    _ weight: UnsafePointer<Float>,
    _ v: UnsafePointer<Float>,
    _ g: UnsafePointer<Float>,
    _ refractory: UnsafePointer<Int16>,
    _ previousDrive: UnsafePointer<Float>,
    _ spikeQueue: UnsafePointer<Int32>,
    _ queueCount: UnsafePointer<Int32>,
    _ flags: UnsafePointer<UInt8>,
    _ last: UnsafePointer<Int64>,
    _ kcMask: UnsafePointer<UInt8>,
    _ modulationMask: UnsafePointer<UInt8>,
    _ rest: UnsafePointer<Float>,
    _ adaptation: UnsafePointer<Float>,
    _ modulation: UnsafePointer<Float>,
    _ modulationLast: UnsafePointer<Int64>
) -> UInt64 {
    do {
        let runtime = try Runtime(
            n: Int(n), edgeCount: Int(edgeCount), slots: Int(slots), clock: clock,
            ptr: ptr, post: post, weight: weight, v: v, g: g,
            refractory: refractory, previousDrive: previousDrive,
            spikeQueue: spikeQueue, queueCount: queueCount, flags: flags,
            last: last, kcMask: kcMask, modulationMask: modulationMask,
            rest: rest, adaptation: adaptation, modulation: modulation,
            modulationLast: modulationLast
        )
        runtimeLock.lock()
        let handle = nextHandle
        nextHandle += 1
        runtimes[handle] = runtime
        runtimeLock.unlock()
        return handle
    } catch {
        recordError(error)
        return 0
    }
}

@_cdecl("fc_destroy")
public func fcDestroy(_ handle: UInt64) {
    runtimeLock.lock()
    runtimes.removeValue(forKey: handle)
    runtimeLock.unlock()
}

@_cdecl("fc_advance")
public func fcAdvance(
    _ handle: UInt64,
    _ drive: UnsafePointer<Float>,
    _ ticks: Int32,
    _ counts: UnsafeMutablePointer<Int32>,
    _ eventIndices: UnsafeMutablePointer<Int32>,
    _ eventClocks: UnsafeMutablePointer<Int64>,
    _ eventCount: UnsafeMutablePointer<Int32>
) -> Double {
    do {
        return try withRuntime(handle) {
            try $0.advance(
                drive: drive, ticks: Int(ticks), outputCounts: counts,
                outputEventIndices: eventIndices, outputEventClocks: eventClocks,
                outputEventCount: eventCount
            )
        }
    } catch {
        recordError(error)
        return -1
    }
}

@_cdecl("fc_read_state")
public func fcReadState(
    _ handle: UInt64,
    _ v: UnsafeMutablePointer<Float>,
    _ g: UnsafeMutablePointer<Float>,
    _ refractory: UnsafeMutablePointer<Int16>,
    _ previousDrive: UnsafeMutablePointer<Float>,
    _ spikeQueue: UnsafeMutablePointer<Int32>,
    _ queueCount: UnsafeMutablePointer<Int32>,
    _ flags: UnsafeMutablePointer<UInt8>,
    _ last: UnsafeMutablePointer<Int64>,
    _ adaptation: UnsafeMutablePointer<Float>,
    _ modulation: UnsafeMutablePointer<Float>,
    _ modulationLast: UnsafeMutablePointer<Int64>
) -> Int32 {
    do {
        try withRuntime(handle) {
            $0.readState(
                v: v, g: g, refractory: refractory, previousDrive: previousDrive,
                spikeQueue: spikeQueue, queueCount: queueCount, flags: flags,
                last: last, adaptation: adaptation, modulation: modulation,
                modulationLast: modulationLast
            )
        }
        return 0
    } catch {
        recordError(error)
        return -1
    }
}

@_cdecl("fc_write_state")
public func fcWriteState(
    _ handle: UInt64,
    _ clock: Int64,
    _ v: UnsafePointer<Float>,
    _ g: UnsafePointer<Float>,
    _ refractory: UnsafePointer<Int16>,
    _ previousDrive: UnsafePointer<Float>,
    _ spikeQueue: UnsafePointer<Int32>,
    _ queueCount: UnsafePointer<Int32>,
    _ flags: UnsafePointer<UInt8>,
    _ last: UnsafePointer<Int64>,
    _ adaptation: UnsafePointer<Float>,
    _ modulation: UnsafePointer<Float>,
    _ modulationLast: UnsafePointer<Int64>
) -> Int32 {
    do {
        try withRuntime(handle) {
            $0.writeState(
                clock: clock, v: v, g: g, refractory: refractory,
                previousDrive: previousDrive, spikeQueue: spikeQueue,
                queueCount: queueCount, flags: flags, last: last,
                adaptation: adaptation, modulation: modulation,
                modulationLast: modulationLast
            )
        }
        return 0
    } catch {
        recordError(error)
        return -1
    }
}

@_cdecl("fc_write_weights")
public func fcWriteWeights(
    _ handle: UInt64,
    _ weights: UnsafePointer<Float>,
    _ count: Int64
) -> Int32 {
    do {
        try withRuntime(handle) { try $0.writeWeights(weights, count: Int(count)) }
        return 0
    } catch {
        recordError(error)
        return -1
    }
}

@_cdecl("fc_update_weights")
public func fcUpdateWeights(
    _ handle: UInt64,
    _ edges: UnsafePointer<Int64>,
    _ weights: UnsafePointer<Float>,
    _ count: Int32
) -> Int32 {
    do {
        try withRuntime(handle) {
            try $0.updateWeights(edges: edges, values: weights, count: Int(count))
        }
        return 0
    } catch {
        recordError(error)
        return -1
    }
}
