#ifndef __ITEM_H__
#define __ITEM_H__

#include <iostream>
#include <string>
#include <vector>
#define PRECISION 10000

class Item;

using itemName = std::string;
using Transaction = std::vector<Item>;


class Item
{
private:

public:
    int myLabel;
    std::string myName;
    int myExternalUtil;
    std::string mySector;
    Item(std::string name, int extUtil, std::string sector);
    const std::string toString() const;
    bool operator<(const Item& i2) const;
    bool operator==(const Item& i2) const;
    ~Item();
};


#endif // __ITEM_H__