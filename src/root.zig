const std = @import("std");
const testing = std.testing;
const Matrix = @import("matrix.zig").Matrix;
const NeuralNetwork = @import("neuralnet.zig").NeuralNetwork;

const nn = NeuralNetwork(3, [_]usize{ 3, 5, 4 });

test "basic matrix init" {
    var m1 = Matrix(1, 1, f64).init(0);
    try testing.expect(m1.values[0][0] == 0);
    m1 = Matrix(1, 1, f64).initwith([1][1]f64{[1]f64{5}});
    try testing.expect(m1.values[0][0] == 5);
}

test "basic matrix multiplication" {
    const v1: [2][3]f64 = [2][3]f64{
        [_]f64{ 1, 2, 3 },
        [_]f64{ 4, 5, 6 },
    };
    var m1 = Matrix(2, 3, f64).initwith(v1);
    const v2: [3][2]f64 = [3][2]f64{
        [_]f64{ 7, 8 },
        [_]f64{ 9, 10 },
        [_]f64{ 11, 12 },
    };
    const m2 = Matrix(3, 2, f64).initwith(v2);
    const m3 = m1.multiplyMatrix(m2);
    //std.debug.print("| {}  {} |\n|{} {} |\n", .{ m3.get(0, 0), m3.get(0, 1), m3.get(1, 0), m3.get(1, 1) });
    try testing.expectEqual(m3.values[0][0], 58);
    try testing.expectEqual(m3.values[0][1], 64);
    try testing.expectEqual(m3.values[1][0], 139);
    try testing.expectEqual(m3.values[1][1], 154);
}

test "strassen matrix multiplication" {
    const v1: [2][2]f64 = [2][2]f64{
        [_]f64{ 1, 2 },
        [_]f64{ 4, 5 },
    };
    var m1 = Matrix(2, 2, f64).initwith(v1);
    const v2: [2][2]f64 = [2][2]f64{
        [_]f64{ 7, 8 },
        [_]f64{ 9, 10 },
    };
    const m2 = Matrix(2, 2, f64).initwith(v2);
    const m3 = m1.multiplyMatrix(m2);
    //std.debug.print("| {}  {} |\n|{} {} |\n", .{ m3.get(0, 0), m3.get(0, 1), m3.get(1, 0), m3.get(1, 1) });
    try testing.expectEqual(m3.values[0][0], 25);
    try testing.expectEqual(m3.values[0][1], 28);
    try testing.expectEqual(m3.values[1][0], 73);
    try testing.expectEqual(m3.values[1][1], 82);
}

test "split strassen matrix multiplication" {
    const v1: [4][4]f64 = [4][4]f64{
        [_]f64{ 1, 2, 3, 4 },
        [_]f64{ 4, 5, 6, 7 },
        [_]f64{ 1, 2, 3, 4 },
        [_]f64{ 4, 5, 6, 7 },
    };
    var m1 = Matrix(4, 4, f64).initwith(v1);
    const v2: [4][4]f64 = [4][4]f64{
        [_]f64{ 7, 8, 11, 12 },
        [_]f64{ 9, 10, 13, 14 },
        [_]f64{ 7, 8, 11, 12 },
        [_]f64{ 9, 10, 13, 14 },
    };
    const m2 = Matrix(4, 4, f64).initwith(v2);
    const m3 = m1.multiplyMatrix(m2);
    //std.debug.print("| {}  {} |\n|{} {} |\n", .{ m3.get(0, 0), m3.get(0, 1), m3.get(1, 0), m3.get(1, 1) });
    try testing.expectEqual(m3.values[0][0], 82);
    try testing.expectEqual(m3.values[0][1], 92);
    try testing.expectEqual(m3.values[0][2], 122);
    try testing.expectEqual(m3.values[0][3], 132);
    try testing.expectEqual(m3.values[1][0], 178);
    try testing.expectEqual(m3.values[1][1], 200);
    try testing.expectEqual(m3.values[1][2], 266);
    try testing.expectEqual(m3.values[1][3], 288);
    try testing.expectEqual(m3.values[2][0], 82);
    try testing.expectEqual(m3.values[2][1], 92);
    try testing.expectEqual(m3.values[2][2], 122);
    try testing.expectEqual(m3.values[2][3], 132);
    try testing.expectEqual(m3.values[3][0], 178);
    try testing.expectEqual(m3.values[3][1], 200);
    try testing.expectEqual(m3.values[3][2], 266);
    try testing.expectEqual(m3.values[3][3], 288);
}

test "vector matrix multiplacation" {
    const v1 = [_]f64{ 1, 2, 3 };
    const v2 = [3][3]f64{
        [_]f64{ 4, 5, 6 },
        [_]f64{ 7, 8, 9 },
        [_]f64{ 10, 11, 12 },
    };
    const m1 = Matrix(1, 3, f64).initwith(.{v1});
    const m2 = Matrix(3, 3, f64).initwith(v2);
    const m3 = m1.multiplyMatrix(m2);
    const v3 = m3.values[0];
    try testing.expectEqual(v3[0], 48);
    try testing.expectEqual(v3[1], 54);
    try testing.expectEqual(v3[2], 60);
}

test "scalar operations" {
    var m1 = Matrix(2, 2, i32).init(0.0);

    m1 = m1.addScalar(1);
    try testing.expectEqual(m1.values[0][0], 1);
    try testing.expectEqual(m1.values[0][1], 1);
    try testing.expectEqual(m1.values[1][0], 1);
    try testing.expectEqual(m1.values[1][1], 1);

    m1 = m1.multiplyScalar(2);
    try testing.expectEqual(m1.values[0][0], 2);
    try testing.expectEqual(m1.values[0][1], 2);
    try testing.expectEqual(m1.values[1][0], 2);
    try testing.expectEqual(m1.values[1][1], 2);
}

fn badMatrixMultiplication(m1: anytype, m2: anytype) Matrix(@TypeOf(m1).m, @TypeOf(m2).n, f64) {
    comptime std.debug.assert(@TypeOf(m2).m == @TypeOf(m1).n);
    var m3 = Matrix(@TypeOf(m1).m, @TypeOf(m2).n, f64).init(0.0);
    for (0..@TypeOf(m1).m) |x| {
        for (0..@TypeOf(m1).n) |y| {
            for (0..@TypeOf(m2).n) |z| {
                m3.values[x][z] = m1.values[x][y] * m2.values[y][z];
            }
        }
    }
    return m3;
}

test "matrix multiplication benchmark" {
    const m1 = Matrix(500, 500, f64).init(12.34);
    const m2 = Matrix(500, 500, f64).init(12.35);
    var time1 = std.time.nanoTimestamp();
    _ = m1.multiplyMatrix(m2);
    time1 -= std.time.nanoTimestamp();
    time1 *= -1;
    var time2 = std.time.nanoTimestamp();
    _ = badMatrixMultiplication(m1, m2);
    time2 -= std.time.nanoTimestamp();
    time2 *= -1;
    //std.debug.print("time1: {any}\ntime2: {any}\n", .{ time1, time2 });
}
