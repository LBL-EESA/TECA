#include "teca_shape_file_util.h"

#include <shapefil.h>

namespace teca_shape_file_util
{

// **************************************************************************
const char *shape_type_name(int shpt)
{
    switch (shpt)
    {
        case SHPT_POINT:
            return "SHPT_POINT";
            break;

        case SHPT_ARC:
            return "SHPT_ARC";
            break;

        case SHPT_POLYGON:
            return "SHPT_POLYGON";
            break;

        case SHPT_MULTIPOINT:
            return "SHPT_MULTIPOINT";
            break;

        case SHPT_POINTZ:
            return "SHPT_POINTZ";
            break;

        case SHPT_ARCZ:
            return "SHPT_ARCZ";
            break;

        case SHPT_POLYGONZ:
            return "SHPT_POLYGONZ";
            break;

        case SHPT_MULTIPOINTZ:
            return "SHPT_MULTIPOINTZ";
            break;

        case SHPT_POINTM:
            return "SHPT_POINTM";
            break;

        case SHPT_ARCM:
            return "SHPT_ARCM";
            break;

        case SHPT_POLYGONM:
            return "SHPT_POLYGONM";
            break;

        case SHPT_MULTIPOINTM:
            return "SHPT_MULTIPOINTM";
            break;

        case SHPT_MULTIPATCH:
            return "SHPT_MULTIPATCH";
            break;
    }

    return "unknown";
}

// **************************************************************************
std::ostream &operator<<(std::ostream &os, const SHPObject &obj)
{
    os << "{" << std::endl
        << "    nSHPType = " << shape_type_name(obj.nSHPType) << std::endl
        << "    nShapeId = " << obj.nShapeId << std::endl
        << "    nParts = " << obj.nParts << std::endl
        << "    panPartStart = ";

    for (int i = 0; i < obj.nParts; ++i)
        os << obj.panPartStart[i] << ", ";
    os << std::endl;

    os << "    panPartType = ";
    for (int i = 0; i < obj.nParts; ++i)
        os << shape_type_name(obj.panPartType[i]) << ", ";
    os << std::endl;

    os << "    nVertices = " << obj.nVertices;
    for (int i = 0; i < obj.nVertices; ++i)
    {
        os << "(" << obj.padfX[i] << ", " << obj.padfY[i] << ", "
            << obj.padfZ[i] << ", " << obj.padfM[i] << "), ";

        if (i % 3 == 0)
            os << std::endl << "        ";
    }
    os << std::endl;

    os << "    bounds = [" << obj.dfXMin << ", " << obj.dfXMax << ", "
        << obj.dfYMin << ", " << obj.dfYMax << ", " << obj.dfZMin << ", "
        << obj.dfZMax << "]" << std::endl
        << "}";

    return os;
}

};
