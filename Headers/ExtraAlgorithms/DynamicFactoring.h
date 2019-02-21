//
//  DynamicFactoring.h
//  FileHFM
//
//  Created by Jean-Marie Mirebeau on 22/08/2018.
//

#ifndef Factoring_h
#define Factoring_h

#include "Base/HamiltonFastMarching.h"

/**
 This class implements dynamic factoring, a technique used to mitigate the impact of the singularities of the solution on the accuracy of the numerical scheme.
 Such singularities arise
 - at the seed points.
 - behind obstacle corners.
 
 While the improvement obtained from including the seed points is obvious, the usefulness of treating the obstacles corners is less clear. Indeed, in the later case, a real improvement is only seen if the obstacles boundaries are aligned with the grid.

 */

/*
 In some rare edge cases, factorization could be slightly less effective than anticipated due to implementation:
- Seed on the boundary of a domain equipped with periodic boundary conditions. (Additional h^2 error)
- Eulerian models with wide initialization around seed, used with dynamic factoring. (Additional h or h^2 error)
 
*/

// TODO : Complete static case. (Setup, SetupIndexStatic)
// TODO ? boolean set to false if no guess made, to speed up in useless case
// TODO : first order differentiation w.r.t. position in third order correction

enum class FactoringPointChoice {Key, Current, Both};

template<typename T>
struct Factoring {
    typedef T Traits;
    typedef HamiltonFastMarching<Traits> HFM;
    Redeclare6Types(HFM,IndexCRef,FullIndexCRef,DiscreteFlowType,OffsetType,ScalarType,DiscreteType)
    Redeclare6Types(HFM,IndexType,IndexDiff,DomainTransformType,VectorType,HFMI,PointType)
    Redeclare3Types(HFM,DistanceGuess,DiscreteFlowElement,DomainType)
    Redeclare1Constant(HFM,Dimension)
    template<typename E, size_t n> using Array = typename Traits::template Array<E,n>;
	
	bool NeedsRecompute(IndexCRef) const;
	bool SetIndexStatic(IndexCRef);
    bool SetIndexDynamic(IndexCRef,const DiscreteFlowType &);
    ScalarType Correction(const OffsetType &, int) const;
	
	std::vector<std::pair<PointType,DistanceGuess> > factoringCenters; // Adimensionized quantities
    ScalarType factoringRadius = 10; // Adimensionized quantity
    Array<bool, Dimension> factoringRegion;
    FactoringPointChoice pointChoice = FactoringPointChoice::Key;
	FactoringMethod	method = FactoringMethod::None;
    bool Setup(HFMI *);
protected:
    const HFM * pFM=nullptr;
	
	// Dynamic factoring only
	std::map<DiscreteType,int> factoringKeypoints;
    Array<bool, Dimension> factoringDone;
    std::multimap<DiscreteType,std::pair<OffsetType,ScalarType> > factoringNeighbors;
    std::vector<std::pair<OffsetType,ScalarType> > currentNeighbors, currentNeighbors2; // cache
	
    struct ElementaryGuess;
    std::vector<ElementaryGuess> guesses;
    void MakeFactor(FullIndexCRef,const DiscreteFlowType &);
    bool MakeGuess(FullIndexCRef);
    
    // Used at setup only
    void SetupCenters(HFMI*);
	void SetupDijkstra();
	void SetupRegion(HFMI*);
    std::vector<std::pair<IndexDiff,ScalarType> > edges;
    bool OnBoundary(IndexCRef,const DomainType &) const;
};

template<typename T>
struct Factoring<T>::ElementaryGuess {
    DistanceGuess guess;
    DomainTransformType transform;
    OffsetType base;
    ScalarType weight;

    VectorType Pull(OffsetType offset) const {
        transform.PullVector(offset); return VectorType::CastCoordinates(offset);}
    ScalarType Value(const OffsetType & offset) const {
		return base==offset ? ScalarType(0.) : weight*guess.Norm(Pull(base-offset) );}
    ScalarType Deriv(const OffsetType & offset) const {
        return -weight*guess.Gradient(Pull(base)).ScalarProduct(Pull(offset));}
    ScalarType Correction(const OffsetType & offset, int ord) const {
        const OffsetType zero = OffsetType::Constant(0);
		switch (ord) {
			case 1: return Deriv(offset) + (Value(zero)-Value(offset));
			case 2: return Deriv(offset) + (1.5*Value(zero) - 2.*Value(offset) + 0.5*Value(2*offset));
			case 3: assert(false);
				return Deriv(offset) + ((11./6.)*Value(zero)-3.*Value(offset)+1.5*Value(2*offset)-(1./3.)*Value(3*offset));
			default: assert(false); return -Traits::Infinity();
		}
    }
    PrintSelfMacro(ElementaryGuess)
};

#include "Implementation/DynamicFactoring.hxx"

#endif /* DynamicFactoring_h */
