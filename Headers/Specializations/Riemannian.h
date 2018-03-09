// HamiltonFastMarching - A Fast-Marching solver with adaptive stencils.
// Copyright (C) 2017 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay.
// Licence GPU GPL v3 or later, see <http://www.gnu.org/licenses/>. Distributed WITHOUT ANY WARRANTY.

#include "CommonTraits.h"
#include "Base/HFMInterface.h"

#ifdef RiemannianHigh // Applies only to dimensions 4 and 5
#include <type_traits> // std::conditional
#include "LinearAlgebra/VoronoiReduction.h"
#endif

// ------------- 2D - 3D Riemannian metrics ------------
template<size_t VDimension>
struct TraitsRiemann : TraitsBase<VDimension> {
    typedef TraitsBase<VDimension> Superclass;
    Redeclare1Type(FromSuperclass,DiscreteType)
    Redeclare1Constant(FromSuperclass,Dimension)

    typedef typename Superclass::template Difference<0> DifferenceType;
    constexpr static const Boundary_AllClosed  boundaryConditions{};
    static const DiscreteType nSymmetric = (Dimension*(Dimension+1))/2;
};
// Linker wants the following line for some obscure reason.
template<size_t VD> constexpr const Boundary_AllClosed TraitsRiemann<VD>::boundaryConditions;

template<size_t VDimension>
struct StencilRiemann : HamiltonFastMarching<TraitsRiemann<VDimension> >::StencilDataType {
    typedef HamiltonFastMarching<TraitsRiemann<VDimension> > HFM;
    typedef typename HFM::StencilDataType Superclass;
    Redeclare7Types(FromHFM,ParamDefault,IndexType,StencilType,ParamInterface,HFMI,Traits,ScalarType)
    Redeclare1Constant(FromHFM,Dimension)
    ParamDefault param;
    
#ifdef RiemannianHigh
    typedef typename Traits::template BasisReduction<Dimension> ReductionType23;
    typedef VoronoiFirstReduction<ScalarType,Dimension> ReductionType45;
    typedef typename std::conditional<Dimension<=3, ReductionType23, ReductionType45>::type ReductionType;
#else
    typedef typename Traits::template BasisReduction<Dimension> ReductionType;
#endif
    
    typedef typename ReductionType::SymmetricMatrixType SymmetricMatrixType;
    typedef SymmetricMatrixType MetricElementType;
    typedef typename Traits::template DataSource<MetricElementType> MetricType;
    std::unique_ptr<MetricType> pDualMetric;
    std::unique_ptr<MetricType> pMetric;
    
    virtual void SetStencil(const IndexType & index, StencilType & stencil) override {
        Voronoi1Mat<ReductionType>(&stencil.symmetric[0],
                                   pDualMetric ? (*pDualMetric)(index) : (*pMetric)(index).Inverse() );
        const ScalarType hm2 = 1/square(param.gridScale);
        for(auto & diff : stencil.symmetric) diff.baseWeight*=hm2;
    }
    virtual const ParamInterface & Param() const override {return param;}
    virtual void Setup(HFMI *that) override {
        Superclass::Setup(that);
        param.Setup(that);
        if(that->io.HasField("dualMetric")) pDualMetric = that->template GetField<MetricElementType>("dualMetric");
        else pMetric = that->template GetField<MetricElementType>("metric");
    }
};

/*
struct TraitsRiemann2 : TraitsBase<2> {
    typedef Difference<0> DifferenceType;
    constexpr static const std::array<Boundary, Dimension>  boundaryConditions =
    {{Boundary::Closed, Boundary::Closed}};
    static const DiscreteType nSymmetric = 3;
};
// Linker wants the following line for some obscure reason.
constexpr const decltype(TraitsRiemann2::boundaryConditions) TraitsRiemann2::boundaryConditions;


struct StencilRiemann2 : HamiltonFastMarching<TraitsRiemann2>::StencilDataType {
    typedef HamiltonFastMarching<TraitsRiemann2> HFM;
    typedef HFM::StencilDataType Superclass;
    HFM::ParamDefault param;
    typedef Traits::BasisReduction<2> ReductionType;
    typedef ReductionType::SymmetricMatrixType SymmetricMatrixType;
    typedef SymmetricMatrixType MetricElementType;
    typedef Traits::DataSource<MetricElementType> MetricType;
    std::unique_ptr<MetricType> pDualMetric;
    std::unique_ptr<MetricType> pMetric;
    
    virtual void SetStencil(const IndexType & index, StencilType & stencil) override {
        Voronoi1Mat<ReductionType>(&stencil.symmetric[0],
                                   pDualMetric ? (*pDualMetric)(index) : (*pMetric)(index).Inverse() );
        const ScalarType hm2 = 1/square(param.gridScale);
        for(auto & diff : stencil.symmetric) diff.baseWeight*=hm2;
    }
    virtual const ParamInterface & Param() const override {return param;}
    virtual void Setup(HFMI *that) override {
        Superclass::Setup(that);
        param.Setup(that);
        if(that->io.HasField("dualMetric")) pDualMetric = that->GetField<MetricElementType>("dualMetric");
        else pMetric = that->GetField<MetricElementType>("metric");
    }
};

// --------------- 3D Riemannian metrics ------------
struct TraitsRiemann3 : TraitsBase<3> {
    typedef Difference<0> DifferenceType;
    constexpr static std::array<Boundary, Dimension>  boundaryConditions =
    {{Boundary::Closed, Boundary::Closed, Boundary::Closed}};
    static const DiscreteType nSymmetric = 6;
};
constexpr decltype(TraitsRiemann3::boundaryConditions) TraitsRiemann3::boundaryConditions;

struct StencilRiemann3 : HamiltonFastMarching<TraitsRiemann3>::StencilDataType {
    typedef HamiltonFastMarching<TraitsRiemann3> HFM;
    typedef HFM::StencilDataType Superclass;
    HFM::ParamDefault param;
    typedef Traits::BasisReduction<3> ReductionType;
    typedef ReductionType::SymmetricMatrixType SymmetricMatrixType;
    typedef SymmetricMatrixType MetricElementType;
    typedef Traits::DataSource<MetricElementType> MetricType;
    std::unique_ptr<MetricType> pDualMetric;
    std::unique_ptr<MetricType> pMetric;

    virtual void SetStencil(const IndexType & index, StencilType & stencil) override {
        Voronoi1Mat<ReductionType>(&stencil.symmetric[0],
                                   pDualMetric ? (*pDualMetric)(index) : (*pMetric)(index).Inverse() );
        const ScalarType hm2 = 1/square(param.gridScale);
        for(auto & diff : stencil.symmetric) diff.baseWeight*=hm2;
    }
    virtual const ParamInterface & Param() const override {return param;}
    virtual void Setup(HFMI *that) override {
        Superclass::Setup(that);
        param.Setup(that);
        if(that->io.HasField("dualMetric")) pDualMetric = that->GetField<MetricElementType>("dualMetric");
        else pMetric = that->GetField<MetricElementType>("metric");
    }};
*/