function init_PEPO(T, β, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    exp_H = exp(-β*onesite_op)
    exp_H_perm = permute(exp_H, ((1,2),()))
    Isom = permute(isomorphism(T, trivspace' ⊗ trivspace', trivspace' ⊗ trivspace'), ((), (1,2,3,4)))
    A = exp_H_perm * Isom
    return Dict((0,0,0,0) => A)
end

function init_PEPO(T, β, onesite_op::AbstractTensorMap, trivspace)
    I = sectortype(onesite_op)
    return init_PEPO(T, β, trivspace, onesite_op)
end

function get_sum_space(highest, highest_loop, spaces)
    summedspace = SumSpace([spaces(i) for i = 0:highest]...)
    for ind = -1:-1:highest_loop
        summedspace = SumSpace(summedspace..., spaces(ind))
    end
    return summedspace
end

function get_PEPO(T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i = keys(PEPO)]) for dir = 1:4]
    highest_loop = [minimum([i[dir] for i = keys(PEPO)]) for dir = 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = zeros(T, SumSpace(pspace) ⊗ SumSpace(pspace)', ⊗([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj,h,hloop) = zip(conjugated,highest,highest_loop)]...))
    for (key, tens) = PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i,h) = zip(key,highest)]
        O[1,1,key...] = tens
    end
    return O
end