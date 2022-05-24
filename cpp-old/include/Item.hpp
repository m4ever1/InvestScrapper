#ifndef __ITEM_H__
#define __ITEM_H__

#include <iostream>


class Item
{
private:

public:
    int myLabel;
    std::string myName;
    float myExternalUtil;
    // float myInternalUtil;
    std::string mySector;
    Item(std::string name, float extUtil, std::string sector);
    std::string toString();

    ~Item();
};
#endif // __ITEM_H__