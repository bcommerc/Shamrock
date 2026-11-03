// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file EulerTimeDerivativeGasPassiveScalars.cpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/EulerTimeDerivativeGasPassiveScalars.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelEulerTimeDerivativeGasPassiveScalars {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_grad_pscal_gas,

            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_dt_pscal_gas,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            u32 npscal_gas) {

            shambase::DistributedData<u32> pscal_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 pscal_count = block_count * block_size * npscal_gas;
                      return pscal_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_vel, spans_grad_pscal_gas},
                sham::DDMultiRef{spans_dt_pscal_gas},
                pscal_counts,
                [npscal_gas](
                    u32 i,
                    const Tvec *__restrict vel,
                    const Tvec *__restrict grad_pscal_gas,
                    Tscal *__restrict dt_pscal_gas) {
                    u32 cell_id     = i / npscal_gas;
                    dt_pscal_gas[i] = -sham::dot(vel[cell_id], grad_pscal_gas[i]);
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeEulerTimeDerivativeGasPassiveScalars<Tvec>::_impl_evaluate_internal() {
        __shamrock_stack_entry();
        auto edges = get_edges();

        edges.spans_vel.check_sizes(edges.sizes.indexes);
        edges.spans_grad_pscal_gas.check_sizes(edges.sizes.indexes);

        edges.spans_dt_pscal_gas.ensure_sizes(edges.sizes.indexes);

        KernelEulerTimeDerivativeGasPassiveScalars<Tvec>::kernel(
            edges.spans_vel.get_spans(),
            edges.spans_grad_pscal_gas.get_spans(),
            edges.spans_dt_pscal_gas.get_spans(),
            edges.sizes.indexes,
            block_size,
            npscal_gas);
    }

    template<class Tvec>
    std::string NodeEulerTimeDerivativeGasPassiveScalars<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Euler time derivatives of the gas passive scalars

            \begin{align}
            {spans_dt_pscal_gas}_{i,j} &= - {spans_vel}_i \cdot {spans_grad_pscal_gas}_{i,j} \\
            i &\in [0,{sizes} * N_{\rm cell/block}) \\
            j &\in [0,n_{\rm pscal}) \\
            n_{\rm pscal} & = {npscal_gas} \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        replace_edges_tex_symbols(tex);

        shambase::replace_all(tex, "{npscal_gas}", sham::format("{}", npscal_gas));
        shambase::replace_all(tex, "{block_size}", sham::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeEulerTimeDerivativeGasPassiveScalars<f64_3>;
