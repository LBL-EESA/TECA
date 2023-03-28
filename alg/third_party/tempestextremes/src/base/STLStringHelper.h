///////////////////////////////////////////////////////////////////////////////
///
///	\file    STLStringHelper.h
///	\author  Paul Ullrich
///	\version July 26, 2010
///
///	<remarks>
///		Copyright 2000-2010 Paul Ullrich
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
///   Removed some functions (ConcatenateStringVector, ParseVariableList,
///   RemoveWhitespaceInPlace, IsInteger, IsIntegerIndex, ToUpper, ToLower)

#ifndef _STLSTRINGHELPER_H_
#define _STLSTRINGHELPER_H_

#include <string>

///	<summary>
///		This class exposes additional functionality which can be used to
///		supplement the STL string class.
///	</summary>
class STLStringHelper {

///////////////////////////////////////////////////////////////////////////////

private:
STLStringHelper() { }

public:

///////////////////////////////////////////////////////////////////////////////

inline static bool IsFloat(const std::string &str) {
	bool fIsFloat = false;
	bool fHasExponent = false;
	bool fHasDecimal = false;
	for(size_t i = 0; i < str.length(); i++) {
		if ((str[i] < '0') || (str[i] > '9')) {
			if (str[i] == '.') {
				if (fHasDecimal) {
					return false;
				}
				if (fHasExponent) {
					return false;
				}
				fHasDecimal = true;
				continue;
			}
			if (str[i] == 'e') {
				if (fHasExponent) {
					return false;
				}
				fHasExponent = true;
				continue;
			}
			if ((str[i] == '-') || (str[i] == '+')) {
				if (i == 0) {
					continue;
				} else if (str[i-1] == 'e') {
					continue;
				} else {
					return false;
				}
			}
			if (str[i] == 'f') {
				if (i != str.length()-1) {
					return false;
				}
			}
			return false;

		} else {
			fIsFloat = true;
		}
	}
	return fIsFloat;
}

};

#endif

