#ifndef __EQUIVTRANS_H__
#define __EQUIVTRANS_H__

#include <vector>
#include "Item.hpp"

class EquivTrans 
{
public:
    std::vector<Item> myItems;
    float myWeight;

    EquivTrans(const Transaction&, float weight);
   ~EquivTrans();
private:
};


#endif // __EQUIVTRANS_H__