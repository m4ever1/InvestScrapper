#include "EquivTrans.hpp"

EquivTrans::EquivTrans(const Transaction& trans, float weight) : myWeight(weight)
{       
    for(const auto& tran : trans)
    {
        myItemNames.push_back(tran.myName);
    }
}