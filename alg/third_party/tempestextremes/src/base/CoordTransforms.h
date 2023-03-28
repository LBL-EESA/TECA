///////////////////////////////////////////////////////////////////////////////
///
///	\file    CoordTransforms.h
///	\author  Paul Ullrich
///	\version March 14, 2020
///
///	<remarks>
///		Copyright 2000-2014 Paul Ullrich
///
///		This file is distributed as part of the Tempest source code package.
///		Permission is granted to use, copy, modify and distribute this
///		source code and its documentation under the terms of the GNU General
///		Public License.  This software is provided "as is" without express
///		or implied warranty.
///	</remarks>
///
///   Amanda Dufek
///   March 28, 2023
///   Removed some functions (StereographicProjectionInv,
///   StereographicProjection, GreatCircleDistanceXYZ_Deg,
///   GreatCircleDistanceXYZ_Rad, VecTransRLL2DtoXYZ_Rad,
///   XYZtoRLL_Rad, XYZtoRLL_Deg, RLLtoXYZ_Deg, RLLtoXYZ_Rad,
///   GreatCircleDistanceFromChordLength_Rad,
///   ChordLengthFromGreatCircleDistance_Deg, LonRadToStandardRange,
///   LonDegToStandardRange, DegToRad)

#ifndef _COORDTRANSFORMS_H_
#define _COORDTRANSFORMS_H_

///////////////////////////////////////////////////////////////////////////////

#include <cmath>

#include "Exception.h"

///////////////////////////////////////////////////////////////////////////////

///	<summary>
///		Convert radians to degrees.
///	</summary>
inline double RadToDeg(
	double dRad
) {
	return (dRad * 180.0 / M_PI);
}

///////////////////////////////////////////////////////////////////////////////

///	<summary>
///		Calculate an average longitude from two given longitudes (in radians).
///	</summary>
inline double AverageLongitude_Rad(
	double dLonRad1,
	double dLonRad2
) {
	double dDeltaLonRad;
	if (dLonRad2 > dLonRad1) {
		dDeltaLonRad = fmod(dLonRad2 - dLonRad1, 2.0 * M_PI);
		if (dDeltaLonRad > M_PI) {
			dDeltaLonRad = dDeltaLonRad - 2.0 * M_PI;
		}
	} else {
		dDeltaLonRad = - fmod(dLonRad1 - dLonRad2, 2.0 * M_PI);
		if (dDeltaLonRad < -M_PI) {
			dDeltaLonRad = dDeltaLonRad + 2.0 * M_PI;
		}
	}

	double dLonRadAvg = dLonRad1 + 0.5 * dDeltaLonRad;

	if ((dLonRadAvg < 0.0) && (dLonRad1 >= 0.0) && (dLonRad2 >= 0.0)) {
		dLonRadAvg += 2.0 * M_PI;
	}
	if ((dLonRadAvg > 2.0 * M_PI) && (dLonRad1 <= 2.0 * M_PI) && (dLonRad2 <= 2.0 * M_PI)) {
		dLonRadAvg -= 2.0 * M_PI;
	}

	return dLonRadAvg;
}

///////////////////////////////////////////////////////////////////////////////

///	<summary>
///		Calculate the great circle distance (in radians) between two RLL points.
///	</summary>
inline double GreatCircleDistance_Rad(
	double dLonRad1,
	double dLatRad1,
	double dLonRad2,
	double dLatRad2
) {
	double dR =
		sin(dLatRad1) * sin(dLatRad2)
		+ cos(dLatRad1) * cos(dLatRad2) * cos(dLonRad2 - dLonRad1);

	if (dR >= 1.0) {
		dR = 0.0;
	} else if (dR <= -1.0) {
		dR = M_PI;
	} else {
		dR = acos(dR);
	}
	if (dR != dR) {
		_EXCEPTIONT("NaN value detected");
	}

	return dR;
}

///////////////////////////////////////////////////////////////////////////////

///	<summary>
///		Calculate the great circle distance (in degrees) between two RLL points.
///	</summary>
inline double GreatCircleDistance_Deg(
	double dLonRad1,
	double dLatRad1,
	double dLonRad2,
	double dLatRad2
) {
	return
		RadToDeg(
			GreatCircleDistance_Rad(
				dLonRad1,
				dLatRad1,
				dLonRad2,
				dLatRad2));
}

#endif // _COORDTRANSFORMS_H_

