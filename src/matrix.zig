const std = @import("std");

fn max(comptime numtype: type, a: numtype, b: numtype) numtype {
    if (a > b) {
        return a;
    }
    return b;
}

fn getSmallestPowerOfTwo(x: usize) usize {
    var i: usize = 1;
    while (i < x) {
        i *= 2;
    }
    return i;
}

test "getSmallestPowerOfTwo" {
    try std.testing.expectEqual(getSmallestPowerOfTwo(3), 4);
}

pub fn Matrix(comptime _m: comptime_int, comptime _n: comptime_int, comptime numtype: type) type {
    return struct {
        pub const m = _m;
        pub const n = _n;
        values: [m][n]numtype,

        pub inline fn init(value: numtype) @This() {
            return .{
                .values = .{.{value} ** n} ** m, //[h][w]f64{[w]f64{0}}
            };
        }
        pub inline fn initwith(values: [m][n]numtype) @This() {
            return .{ .values = values };
        }

        pub fn addMatrix(m1: @This(), m2: @This()) @This() {
            var m3 = @This().init(0);
            for (0..@TypeOf(m1).m) |x| {
                for (0..@TypeOf(m1).n) |y| {
                    m3.values[x][y] = m1.values[x][y] + m2.values[x][y];
                }
            }
            return m3;
        }
        pub fn subtractMatrix(m1: @This(), m2: @This()) @This() {
            var m3 = @This().init(0);
            for (0..@TypeOf(m1).m) |x| {
                for (0..@TypeOf(m1).n) |y| {
                    m3.values[x][y] = m1.values[x][y] - m2.values[x][y];
                }
            }
            return m3;
        }
        pub fn multiplyMatrix(m1: @This(), m2: anytype) Matrix(m, @TypeOf(m2).n, numtype) {
            comptime std.debug.assert(@TypeOf(m2).m == n);
            const l = @TypeOf(m2).n;
            const s = comptime getSmallestPowerOfTwo(max(comptime_int, max(comptime_int, m, n), l));

            var v1: [s][s]numtype = .{.{0b0} ** s} ** s;
            var v2: [s][s]numtype = .{.{0b0} ** s} ** s;
            var v3: [s][s]numtype = .{.{0b0} ** s} ** s;
            for (0..m) |x| {
                @memcpy(v1[x][0..n], m1.values[x][0..n]);
            }
            for (0..n) |x| {
                @memcpy(v2[x][0..l], m2.values[x][0..l]);
            }
            const A = ZeroCoords(numtype, s, &v1);
            const B = ZeroCoords(numtype, s, &v2);
            const C = ZeroCoords(numtype, s, &v3);
            multiply(numtype, s, s, A, B, C);
            const m3 = Matrix(m, l, numtype).initwith(v3);
            return m3;
        }
        pub fn addScalar(m1: @This(), v2: numtype) @This() {
            var v3 = m1.values;
            for (0..m) |x| {
                for (0..n) |y| {
                    v3[x][y] += v2;
                }
            }
            return @This().initwith(v3);
        }
        pub fn multiplyScalar(m1: @This(), v2: numtype) @This() {
            var v3 = m1.values;
            for (0..m) |x| {
                for (0..n) |y| {
                    v3[x][y] *= v2;
                }
            }
            return @This().initwith(v3);
        }
    };
}

// This block is for taking a square power of 2 set of 2 matricies and multiplying them using the stassen algoritm
// with absolutely minimum memory usage. All additions and subtractions should be in place, and all operations
// should be operating on a single set of 3 matricies (A, B, and C) with the exception of the intermediary matricies
// used in the strassen multiplication.

fn Coordinates(comptime numtype: type, comptime n: comptime_int) type {
    return struct {
        x: usize,
        y: usize,
        ptr: *[n][n]numtype,
    };
}
const AddSubtract = enum {
    Add,
    Subtract,
};
const Add = AddSubtract.Add;
const Subtract = AddSubtract.Subtract;
inline fn ZeroCoords(comptime numtype: type, comptime n: comptime_int, ptr: *[n][n]numtype) Coordinates(numtype, n) {
    return Coordinates(numtype, n){
        .x = 0b0,
        .y = 0b0,
        .ptr = ptr,
    };
}
inline fn inPlaceAddSubtract(
    comptime numtype: type,
    comptime n: comptime_int,
    comptime l: comptime_int,
    Acoordinates: Coordinates(numtype, n),
    Bcoordinates: Coordinates(numtype, n),
    comptime addSubtract: AddSubtract,
) void {
    for (0..l) |x| {
        for (0..l) |y| {
            switch (addSubtract) {
                AddSubtract.Add => {
                    Acoordinates.ptr.*[x + Acoordinates.x][y + Acoordinates.y] +=
                        Bcoordinates.ptr.*[x + Bcoordinates.x][y + Bcoordinates.y];
                },
                AddSubtract.Subtract => {
                    Acoordinates.ptr.*[x + Acoordinates.x][y + Acoordinates.y] -=
                        Bcoordinates.ptr.*[x + Bcoordinates.x][y + Bcoordinates.y];
                },
            }
        }
    }
}
test "Matrix In Place Add and Subtract" {
    var A: [2][2]i32 = .{.{2} ** 2} ** 2;
    var B: [2][2]i32 = .{.{2} ** 2} ** 2;
    try std.testing.expectEqual(2, A[0][0]);
    inPlaceAddSubtract(i32, 2, 2, ZeroCoords(i32, 2, &A), ZeroCoords(i32, 2, &B), Add);
    try std.testing.expectEqual(4, A[0][0]);
    inPlaceAddSubtract(i32, 2, 2, ZeroCoords(i32, 2, &A), ZeroCoords(i32, 2, &B), Subtract);
    try std.testing.expectEqual(2, A[0][0]);
    const Abrcorner = Coordinates(i32, 2){ .x = 1, .y = 1, .ptr = &A };
    const Bbrcorner = Coordinates(i32, 2){ .x = 1, .y = 1, .ptr = &B };
    inPlaceAddSubtract(i32, 2, 1, Abrcorner, Bbrcorner, Add);
    try std.testing.expectEqual(2, A[0][0]);
    try std.testing.expectEqual(4, A[1][1]);
}
inline fn copy(
    comptime numtype: type,
    comptime n: comptime_int,
    comptime l: comptime_int,
    Acoordinates: Coordinates(numtype, n),
    Bcoordinates: Coordinates(numtype, n),
) void {
    for (0..n) |x| {
        @memcpy(Bcoordinates.ptr.*[x * Bcoordinates.x][Bcoordinates.y .. Bcoordinates.y + l], Acoordinates.ptr.*[x * Acoordinates.x][Acoordinates.y .. Acoordinates.y + l]);
    }
}
test "matrix copy" {
    var A: [2][2]i32 = .{.{2} ** 2} ** 2;
    var B: [2][2]i32 = .{.{0} ** 2} ** 2;
    copy(i32, 2, 2, ZeroCoords(i32, 2, &A), ZeroCoords(i32, 2, &B));
    try std.testing.expectEqual(B[0][0], 2);
}
fn multiply(
    comptime numtype: type,
    comptime n: comptime_int, // size of each dimension of the matricies
    comptime l: comptime_int, // size of the sections of the matricies to operate on
    Acoordinates: Coordinates(numtype, n), // coordinates of the A matrix to operate on
    Bcoordinates: Coordinates(numtype, n), // coordinates of the B matrix to operate on
    Ccoordinates: Coordinates(numtype, n), // coordinates of the C matrix to operate on
) void {
    comptime std.debug.assert(n & (n - 1) == 0);
    if (n == 0) {
        // do nothing
    } else if (n == 1) {
        Ccoordinates.ptr.*[0][0] = Acoordinates.ptr.*[0][0] * Bcoordinates.ptr.*[0][0];
    } else {
        const mid = l / 2;
        // coordinates for all matrix quadrinths
        const A11 = Coordinates(numtype, n){ .x = Acoordinates.x, .y = Acoordinates.y, .ptr = Acoordinates.ptr };
        const A12 = Coordinates(numtype, n){ .x = Acoordinates.x, .y = Acoordinates.y + mid, .ptr = Acoordinates.ptr };
        const A21 = Coordinates(numtype, n){ .x = Acoordinates.x + mid, .y = Acoordinates.y, .ptr = Acoordinates.ptr };
        const A22 = Coordinates(numtype, n){ .x = Acoordinates.x + mid, .y = Acoordinates.y + mid, .ptr = Acoordinates.ptr };
        const B11 = Coordinates(numtype, n){ .x = Bcoordinates.x, .y = Bcoordinates.y, .ptr = Bcoordinates.ptr };
        const B12 = Coordinates(numtype, n){ .x = Bcoordinates.x, .y = Bcoordinates.y + mid, .ptr = Bcoordinates.ptr };
        const B21 = Coordinates(numtype, n){ .x = Bcoordinates.x + mid, .y = Bcoordinates.y, .ptr = Bcoordinates.ptr };
        const B22 = Coordinates(numtype, n){ .x = Bcoordinates.x + mid, .y = Bcoordinates.y + mid, .ptr = Bcoordinates.ptr };
        const C11 = Coordinates(numtype, n){ .x = Ccoordinates.x, .y = Ccoordinates.y, .ptr = Ccoordinates.ptr };
        const C12 = Coordinates(numtype, n){ .x = Ccoordinates.x, .y = Ccoordinates.y + mid, .ptr = Ccoordinates.ptr };
        const C21 = Coordinates(numtype, n){ .x = Ccoordinates.x + mid, .y = Ccoordinates.y, .ptr = Ccoordinates.ptr };
        const C22 = Coordinates(numtype, n){ .x = Ccoordinates.x + mid, .y = Ccoordinates.y + mid, .ptr = Ccoordinates.ptr };

        // intermediary step that gets switched out as i need it
        var m = .{.{0b0} ** mid} ** mid;
        const M = ZeroCoords(numtype, n, &m);
        var p = .{.{0b0} ** mid} ** mid;
        const P = ZeroCoords(numtype, n, &p);

        // M1 = (A11 + A22) * (B11 + B22)
        // M2 = (A21 + A22) * B11
        // M3 = A11 * (B12 - B22)
        // M4 = A22 * (B21 - B11)
        // M5 = (A11 + A12) * B22
        // M6 = (A21 - A11) * (B11 + B12)
        // M7 = (A12 - A22) * (B21 + B22)
        //
        // C11 = M1 + M4 - M5 + M7
        // C12 = M3 + M5
        // C21 = M2 + M4
        // C22 = M1 - M2 + M3 + M6

        // Calculate M1 into C11 using P and M, then copy C11 into C22
        copy(numtype, n, mid, P, A11); // copy A11 into P
        inPlaceAddSubtract(numtype, n, mid, P, A22, Add); // add A22 to P
        copy(numtype, n, mid, M, B11); // copy B11 into M
        inPlaceAddSubtract(numtype, n, mid, M, B22, Add); // add B22 to M
        multiply(numtype, n, mid, M, P, C11); // multiply P and M into C11
        copy(numtype, n, mid, C22, C11); // copy C11 into C22

        // Calculate M6 into C12 using M and P and add it to C22
        copy(numtype, n, mid, P, A21); // copy A21 into P
        inPlaceAddSubtract(numtype, n, mid, P, A11, Subtract); // subtract A11 from P
        copy(numtype, n, mid, M, B11); // copy B11 into M
        inPlaceAddSubtract(numtype, n, mid, M, B22, Add); // add B22 to M
        multiply(numtype, n, mid, P, M, C12); // multiply P and M into C12
        inPlaceAddSubtract(numtype, n, mid, C22, C12, Add); // add C12 to C22

        // Calculate M7 into C12 using M and P and add it to C11
        copy(numtype, n, mid, P, A12); // copy A12 into P
        inPlaceAddSubtract(numtype, n, mid, P, A22, Subtract); // subtract A22 from P
        copy(numtype, n, mid, M, B21); // copy B21 into M
        inPlaceAddSubtract(numtype, n, mid, M, B22, Add); // add B22 to M
        multiply(numtype, n, mid, P, M, C12); // multiply P and M into C12
        inPlaceAddSubtract(numtype, n, mid, C11, C12, Add); // add C12 to C11

        // Calculate M2 into C21 using P
        copy(numtype, n, mid, P, A21); // copy A21 into P
        inPlaceAddSubtract(numtype, n, mid, P, A22, Add); // add A22 to P
        multiply(numtype, n, mid, P, B11, C21); // multiply P and B11 into C21

        // Calculate M3 into C12 using P
        copy(numtype, n, mid, P, B12); // copy B12 into P
        inPlaceAddSubtract(numtype, n, mid, P, B22, Subtract); // Subtract B22 from P
        multiply(numtype, n, mid, P, A11, C12); // multiply P and A11 into C21

        // Calculate M4 into M using P, then add it to C11 and C21
        copy(numtype, n, mid, P, B21); // copy B21 into P
        inPlaceAddSubtract(numtype, n, mid, P, B11, Subtract); // Subtract B11 from P
        multiply(numtype, n, mid, P, A22, M); // multiply P and A22 into M
        inPlaceAddSubtract(numtype, n, mid, C11, M, Add); // add M to C11
        inPlaceAddSubtract(numtype, n, mid, C21, M, Add); // add M to C21

        // Calculate M5 into M using P, then add it to C11 and C12
        copy(numtype, n, mid, P, A11); // copy A11 into P
        inPlaceAddSubtract(numtype, n, mid, P, A12, Add); // Subtract A12 from P
        multiply(numtype, n, mid, P, B22, M); // multiply P and B22 into M
        inPlaceAddSubtract(numtype, n, mid, C11, M, Add); // add M to C11
        inPlaceAddSubtract(numtype, n, mid, C12, M, Add); // add M to C12

    }
}
