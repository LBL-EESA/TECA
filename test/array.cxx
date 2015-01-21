#include "array.h"
#include "teca_binary_stream.h"

void array::to_stream(teca_binary_stream &s)
{
    s.pack(this->name);
    s.pack(this->extent);
    s.pack(this->data);
}

void array::from_stream(teca_binary_stream &s)
{
    s.unpack(this->name);
    s.unpack(this->extent);
    s.unpack(this->data);
}
