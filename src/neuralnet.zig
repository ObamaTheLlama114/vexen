const Matrix = @import("matrix.zig");

pub fn NeuralNetwork(comptime layerAmount: comptime_int, comptime layers: [layerAmount]comptime_int) type {
    var matriciesSize = 0;
    for (0..layerAmount - 1) |i| {
        matriciesSize += @sizeOf(Matrix(layers[i], layers[i + 1]));
    }
    matriciesSize /= @sizeOf(f64);
    return struct {};
}
