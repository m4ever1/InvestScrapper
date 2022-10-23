#include <iostream>
#include "Item.hpp"

Item::Item(std::string name, int extUtil, int sector) : 
myName(name), 
myExternalUtil(extUtil), 
// myInternalUtil(intUtil),
mySector(sector)
{
}

const std::string Item::toString() const
{
   return std::string("TODO");
}

Item::~Item()
{
}

bool Item::operator<(const Item& i2) const
{
   return this->myName < i2.myName;
}

bool Item::operator==(const Item& i2) const
{
   return this->myName == i2.myName;
}