#pragma once


class EquivTrans {
    
    std::vector<itemName> itemNames;
    float weight;


public:
    EquivTrans(const Transaction, float weight);
    ~EquivTrans();
};