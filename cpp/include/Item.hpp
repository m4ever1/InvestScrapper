#ifndef __ITEM_H__
#define __ITEM_H__

#include <iostream>
#include <string>

class Item;

using itemName = std::string;
using Transaction = std::vector<Item>;


class Item
{
private:

public:
    int myLabel;
    std::string myName;
    float myExternalUtil;
    std::string mySector;
    Item(std::string name, float extUtil, std::string sector);
    std::string toString();
    bool operator<(const Item& i2) const;
    ~Item();
};


#endif // __ITEM_H__