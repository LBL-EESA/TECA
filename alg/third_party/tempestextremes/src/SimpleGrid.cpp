///////////////////////////////////////////////////////////////////////////////
///
///	\file    SimpleGrid.cpp
///	\author  Paul Ullrich
///	\version August 30, 2019
///
///	<remarks>
///		Copyright 2019 Paul Ullrich
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
///   Removed some functions (HasAreas, HasConnectivity,
///   GenerateLatitudeLongitude, GenerateRegionalLatitudeLongitude,
///   GenerateRectilinearStereographic, FromMeshFV, FromMeshFE, FromFile,
///   ToFile, FromUnstructuredDataFile, DimCount, CoordinateVectorToIndex,
///   BuildKDTree, NearestNode, NearestNodes)
///   and m_kdtree and c_szFileIdentifier variables.

#include "SimpleGrid.h"
#include "CoordTransforms.h"
#include "Announce.h"

///////////////////////////////////////////////////////////////////////////////

SimpleGrid::~SimpleGrid() { }

///////////////////////////////////////////////////////////////////////////////

bool SimpleGrid::IsInitialized() const {
	if ((m_nGridDim.size() != 0) ||
	    m_dLon.IsAttached() ||
	    m_dLat.IsAttached() ||
	    m_dArea.IsAttached() ||
	    (m_vecConnectivity.size() != 0)
	) {
		return true;
	}
	return false;
}

///////////////////////////////////////////////////////////////////////////////

void SimpleGrid::GenerateRectilinearConnectivity(
	int nLat,
	int nLon,
	bool fRegional,
	bool fDiagonalConnectivity
) {
	m_vecConnectivity.clear();
	m_vecConnectivity.resize(nLon * nLat);

	size_t ixs = 0;
	for (int j = 0; j < nLat; j++) {
	for (int i = 0; i < nLon; i++) {

		// Connectivity in eight directions
		if (fDiagonalConnectivity) {
			if (fRegional) {
				for (int ix = -1; ix <= 1; ix++) {
				for (int jx = -1; jx <= 1; jx++) {
					if ((ix == 0) && (jx == 0)) {
						continue;
					}

					int inew = i + ix;
					int jnew = j + jx;

					if ((inew < 0) || (inew >= nLon)) {
						continue;
					}
					if ((jnew < 0) || (jnew >= nLat)) {
						continue;
					}

					m_vecConnectivity[ixs].push_back(jnew * nLon + inew);
				}
				}

			} else {
				for (int ix = -1; ix <= 1; ix++) {
				for (int jx = -1; jx <= 1; jx++) {
					if ((ix == 0) && (jx == 0)) {
						continue;
					}

					int inew = i + ix;
					int jnew = j + jx;

					if ((jnew < 0) || (jnew >= nLat)) {
						continue;
					}
					if (inew < 0) {
						inew += nLon;
					}
					if (inew >= nLon) {
						inew -= nLon;
					}

					m_vecConnectivity[ixs].push_back(jnew * nLon + inew);
				}
				}
			}

		// Connectivity in the four primary directions
		} else {
			if (j != 0) {
				m_vecConnectivity[ixs].push_back((j-1) * nLon + i);
			}
			if (j != nLat-1) {
				m_vecConnectivity[ixs].push_back((j+1) * nLon + i);
			}

			if ((!fRegional) ||
			    ((i != 0) && (i != nLon-1))
			) {
				m_vecConnectivity[ixs].push_back(
					j * nLon + ((i + 1) % nLon));
				m_vecConnectivity[ixs].push_back(
					j * nLon + ((i + nLon - 1) % nLon));
			}
		}

		ixs++;
	}
	}
}

///////////////////////////////////////////////////////////////////////////////

void SimpleGrid::GenerateLatitudeLongitude(
	const DataArray1D<double> & vecLat,
	const DataArray1D<double> & vecLon,
	bool fRegional,
	bool fDiagonalConnectivity,
	bool fVerbose
) {
	if (IsInitialized()) {
		_EXCEPTIONT("Attempting to call GenerateLatitudeLongitude() on previously initialized grid");
	}

	int nLat = vecLat.GetRows();
	int nLon = vecLon.GetRows();

	if (nLat < 2) {
		_EXCEPTIONT("At least two latitudes needed to generate grid.");
	}
	if (nLon < 2) {
		_EXCEPTIONT("At least two longitudes needed to generate grid.");
	}

	m_dLat.Allocate(nLon * nLat);
	m_dLon.Allocate(nLon * nLat);
	m_dArea.Allocate(nLon * nLat);

	m_nGridDim.resize(2);
	m_nGridDim[0] = nLat;
	m_nGridDim[1] = nLon;

	// Verify units of latitude and longitude
	bool fCalculateArea = true;

	for (int j = 0; j < nLat; j++) {
		if (fabs(vecLat[j]) > 0.5 * M_PI + 1.0e-12) {
			if (fRegional) {
				Announce("WARNING: Latitude array out of bounds [-90,90] "
					"(%1.5f); defaulting grid areas to 1.",
					RadToDeg(vecLat[j]));
				fCalculateArea = false;
				break;
			} else {
				_EXCEPTION1("Latitude array out of bounds [-90,90] (%1.5f). "
					 "Did you mean to specify \"--regional\"?",
					 RadToDeg(vecLat[j]));
			}
		}
	}
	for (int i = 0; i < nLon; i++) {
		if (fabs(vecLon[i]) > 3.0 * M_PI + 1.0e-12) {
			if (fRegional) {
				Announce("WARNING: Longitude array out of bounds [-540,540] "
					"(%1.5f); defaulting grid areas to 1.",
					RadToDeg(vecLon[i]));
				fCalculateArea = false;
				break;
			} else {
            Announce("Error lon");
				_EXCEPTION1("Longitude array out of bounds [-540,540] (%1.5f). "
					"Did you mean to specify \"--regional\"?",
					RadToDeg(vecLon[i]));
			}
		}
	}

   Announce("Determine orientation of latitude array");

	// Determine orientation of latitude array
	double dLatOrient = 1.0;
	if (vecLat[1] < vecLat[0]) {
		dLatOrient = -1.0;
	}
	for (int j = 0; j < nLat-1; j++) {
		if (dLatOrient * vecLat[1] < dLatOrient * vecLat[0]) {
			_EXCEPTIONT("Latitude array must be monotone.");
		}
	}

	int ixs = 0;
	for (int j = 0; j < nLat; j++) {
	for (int i = 0; i < nLon; i++) {

		// Vectorize coordinates
		m_dLat[ixs] = vecLat[j];
		m_dLon[ixs] = vecLon[i];

		// Bounds of each volume and associated area
		double dLatRad1;
		double dLatRad2;

		double dLonRad1;
		double dLonRad2;

		if (j == 0) {
			if (fRegional) {
				dLatRad1 = vecLat[0] - 0.5 * (vecLat[1] - vecLat[0]);
			} else {
				dLatRad1 = - dLatOrient * 0.5 * M_PI;
			}
		} else {
			dLatRad1 = 0.5 * (vecLat[j-1] + vecLat[j]);
		}
		if (j == nLat-1) {
			if (fRegional) {
				dLatRad2 = vecLat[j] + 0.5 * (vecLat[j] - vecLat[j-1]);
			} else {
				dLatRad2 = dLatOrient * 0.5 * M_PI;
			}
		} else {
			dLatRad2 = 0.5 * (vecLat[j+1] + vecLat[j]);
		}

		if (i == 0) {
			if (fRegional) {
				dLonRad1 = vecLon[0] - 0.5 * (vecLon[1] - vecLon[0]);
			} else {
				dLonRad1 = AverageLongitude_Rad(vecLon[0], vecLon[nLon-1]);
			}
		} else {
			dLonRad1 = AverageLongitude_Rad(vecLon[i-1], vecLon[i]);
		}

		if (i == nLon-1) {
			if (fRegional) {
				dLonRad2 = vecLon[i] + 0.5 * (vecLon[i] - vecLon[i-1]);
			} else {
				dLonRad2 = AverageLongitude_Rad(vecLon[nLon-1], vecLon[0]);
			}
		} else {
			dLonRad2 = AverageLongitude_Rad(vecLon[i], vecLon[i+1]);
		}

		// Because of the way AverageLongitude_Rad works,
		if (dLonRad1 > dLonRad2) {
			dLonRad1 -= 2.0 * M_PI;
		}
		double dDeltaLon = dLonRad2 - dLonRad1;
		if (!fRegional && (dDeltaLon >= M_PI)) {
			_EXCEPTION1("Grid element longitudinal extent too large (%1.7f deg).  Did you mean to specify \"--regional\"?",
				dDeltaLon * 180.0 / M_PI);
		}

		if (fCalculateArea) {
			m_dArea[ixs] = fabs(sin(dLatRad2) - sin(dLatRad1)) * dDeltaLon;
		} else {
			m_dArea[ixs] = 1.0;
		}

		ixs++;
	}
	}

   Announce("GenerateRectilinearConnectivity");

	// Generate connectivity
	GenerateRectilinearConnectivity(nLat, nLon, fRegional, fDiagonalConnectivity);

	// Output total area
	{
		double dTotalArea = 0.0;
		for (size_t i = 0; i < m_dArea.GetRows(); i++) {
			dTotalArea += m_dArea[i];
		}
		if (fVerbose) {
			Announce("Total calculated grid area: %1.15e sr", dTotalArea);
		}
	}

}
