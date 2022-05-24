#pragma once
#include <vector>
#include "Item.hpp"

class EquivTrans {
    
    std::vector<itemName> myItemNames;
    float myWeight;


public:
    EquivTrans(const Transaction&, float weight);
   ~EquivTrans();
};