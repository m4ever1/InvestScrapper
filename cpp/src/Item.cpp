#include <iostream>
#include "Item.hpp"

Item::Item(std::string name, float extUtil, std::string sector) : 
myName(name), 
myExternalUtil(extUtil), 
// myInternalUtil(intUtil),
mySector(sector)
{
}

Item::~Item()
{
}

bool Item::operator<(const Item& i2) const
{
   return this->myName < i2.myName;
}