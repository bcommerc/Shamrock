// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrimGasPassiveScalars.cpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ConsToPrimGasPassiveScalars.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelConsToPrimGasPassiveScalars {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_gas_pscal,

            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_pscal_gas,
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
                sham::DDMultiRef{spans_rho, spans_rho_gas_pscal},
                sham::DDMultiRef{spans_pscal_gas},
                pscal_counts,
                [npscal_gas](
                    u32 i,
                    const Tscal *__restrict rho,
                    const Tscal *__restrict rho_gas_pscal,
                    Tscal *__restrict pscal_gas) {
                    u32 cell_id  = i / npscal_gas;
                    pscal_gas[i] = rho_gas_pscal[i] / rho[cell_id];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeConsToPrimGasPassiveScalars<Tvec>::_impl_evaluate_internal() {
        __shamrock_stack_entry();
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_rho_gas_pscal.check_sizes(edges.sizes.indexes);

        edges.spans_pscal_gas.ensure_sizes(edges.sizes.indexes);

        KernelConsToPrimGasPassiveScalars<Tvec>::kernel(
            edges.spans_rho.get_spans(),
            edges.spans_rho_gas_pscal.get_spans(),
            edges.spans_pscal_gas.get_spans(),
            edges.sizes.indexes,
            block_size,
            npscal_gas);
    }

    template<class Tvec>
    std::string NodeConsToPrimGasPassiveScalars<Tvec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Conservative to primitive variable (gas passive scalars)

            \begin{align}
            {spans_pscal_gas}_{i,j} &= \frac{ {spans_rho_gas_pscal}_{i,j} }{ {spans_rho}_i } \\
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

template class shammodels::basegodunov::modules::NodeConsToPrimGasPassiveScalars<f64_3>;
