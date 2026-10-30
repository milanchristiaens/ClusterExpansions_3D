struct Cubic <: Lattice end

# QR decomposition
function QR_two_pepo_left(::Cubic, O1, O2, ind)
    pb = (1, ind)
    O1p = @timeit to "QR left: permute O1" permute(O1, (Tuple(setdiff(1:8, pb)), pb))
    _, Rb = @timeit to "QR left: left_orth O1" left_orth(O1p)
    pt = (2, ind)
    O2p = @timeit to "QR left: permute O2" permute(O2, (Tuple(setdiff(1:8, pt)), pt))
    _, Rt = @timeit to "QR left: left_orth O2" left_orth(O2p)
    M = @timeit to "QR left: contract Rt*Rb" begin
        @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
        M
    end
    Mp = @timeit to "QR left: permute M" permute(M, ((3, 4), (1, 2)))
    _, R = @timeit to "QR left: left_orth M" left_orth(Mp)
    return R
end

function QR_two_pepo_right(::Cubic, O1, O2, ind)
    pb = (1, ind)
    O1p = @timeit to "QR right: permute O1" permute(O1, (pb, Tuple(setdiff(1:8, pb))))
    Rb, _ = @timeit to "QR right: right_orth O1" right_orth(O1p)
    pt = (2, ind)
    O2p = @timeit to "QR right: permute O2" permute(O2, (pt, Tuple(setdiff(1:8, pt))))
    Rt, _ = @timeit to "QR right: right_orth O2" right_orth(O2p)
    M = @timeit to "QR right: contract Rt*Rb" begin
        @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
        M
    end
    Mp = @timeit to "QR right: permute M" permute(M, ((1, 2), (3, 4)))
    R, _ = @timeit to "QR right: right_orth M" right_orth(Mp)
    return R
end

# Function used by rest of package, it takes the pair (lower layer, upper layer)
# multiplies them and returns one tensor with the original bond dimension together with error estimate
function approximate_state(
        A::Tuple{AbstractTensorMap{E, S, 2, 6}, AbstractTensorMap{E, S, 2, 6}},
        trunc_alg::NoEnvTruncation; rot_invariance::Bool = false
    ) where {E, S <: ElementarySpace}
    proj = rot_invariance ? :takagi : :svd
    lattice = Cubic()
    PU, PD = @timeit to "projectors UP-DOWN (find_P1P2)" find_P1P2(lattice, A, 3, 6, trunc_alg.trscheme; proj)
    # @info "$(norm(PU.data - (PD').data)) / $(norm(PU))"
    # @info "$(norm(PU.data - conj.((PD').data))) / $(norm(PU))"
    if rot_invariance
        PN = PE = copy(PU)
        PS = PW = copy(PD)
    else
        PN, PS = @timeit to "projectors NORTH-SOUTH (find_P1P2)" find_P1P2(lattice, A, 4, 7, trunc_alg.trscheme; proj)
        PE, PW = @timeit to "projectors EAST-WEST (find_P1P2)" find_P1P2(lattice, A, 5, 8, trunc_alg.trscheme; proj)
    end
    Onew = @timeit to "stack and truncate: contraction (A1 A2 P*6)" begin
        @tensor opt = true Onew[-1 -2; -3 -4 -5 -6 -7 -8] := A[1][1 -2; 8 9 10 11 12 13] * A[2][-1 1; 2 3 4 5 6 7] * PU[2 8; -3] * PN[3 9; -4] * PE[4 10; -5] * PD[-6; 5 11] * PS[-7; 6 12] * PW[-8; 7 13]
        Onew 
    end
    return Onew, nothing
end