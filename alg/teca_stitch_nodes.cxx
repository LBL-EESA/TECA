#include "teca_stitch_nodes.h"

#include "teca_variant_array_impl.h"
#include "teca_table.h"

#include <kdtree.h>
#include "Announce.h"
#include "TimeObj.h"
#include "CoordTransforms.h"
#include "STLStringHelper.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#define TECA_DEBUG 2

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
typedef std::vector<std::vector<std::string>> TimesliceCandidateInformation;
typedef std::pair<Time, TimesliceCandidateInformation> TimeToCandidateInfoPair;
typedef std::map<Time, TimesliceCandidateInformation> TimeToCandidateInfoMap;
typedef TimeToCandidateInfoMap::iterator TimeToCandidateInfoMapIterator;

// --------------------------------------------------------------------------
/**
 * Read a DetectNodes output file into mapCandidateInfo
 */
int parse_detect_nodes_file(
    const_p_teca_table &candidates,
    const std::vector<std::string> &vec_format_strings,
    TimeToCandidateInfoMap &mapCandidateInfo,
    Time::CalendarType caltype,
    bool allow_repeated_times)
{
    // Current read state
    enum ReadState
    {
      ReadState_Time,
      ReadState_Candidate
    } eReadState = ReadState_Time;

    // Current candidate at this time
    int iCandidate  = 0;
    // Total candidates at this time
    int nCandidates = 0;

    // Current candidate information
    TimeToCandidateInfoMapIterator iterCurrentTime = mapCandidateInfo.end();

    auto year = candidates->get_column("year");
    auto month = candidates->get_column("month");
    auto day = candidates->get_column("day");
    auto hour = candidates->get_column("hour");
    auto minute = candidates->get_column("minute");
    auto n_candidates = candidates->get_column("ncandidates");
    auto lon = candidates->get_column("lon");
    auto lat = candidates->get_column("lat");
    auto ii = candidates->get_column("i");
    auto jj = candidates->get_column("j");

    unsigned long n_rows = candidates->get_number_of_rows();

    VARIANT_ARRAY_DISPATCH(lat.get(),

       auto [sp_year, p_year] = get_host_accessible<const teca_int_array>(year);
       auto [sp_month, p_month] = get_host_accessible<const teca_int_array>(month);
       auto [sp_day, p_day] = get_host_accessible<const teca_int_array>(day);
       auto [sp_hour, p_hour] = get_host_accessible<const teca_int_array>(hour);
       auto [sp_minute, p_minute] = get_host_accessible<const teca_int_array>(minute);
       auto [sp_n_candidates, p_n_candidates] = get_host_accessible<const teca_long_array>(n_candidates);
       auto [sp_lat, p_lat] = get_host_accessible<CTT>(lat);
       auto [sp_lon, p_lon] = get_host_accessible<CTT>(lon);
       auto [sp_ii, p_ii] = get_host_accessible<const teca_int_array>(ii);
       auto [sp_jj, p_jj] = get_host_accessible<const teca_int_array>(jj);

       for (unsigned long row = 0; row < n_rows; ++row)
       {
          // Parse the time
          if (eReadState == ReadState_Time)
          {
             iCandidate = 0;
             nCandidates = p_n_candidates[row];

             Time time(caltype);
             time.SetYear(p_year[row]);
             time.SetMonth(p_month[row]);
             time.SetDay(p_day[row]);
             time.SetSecond(p_hour[row] * 3600 + p_minute[row] * 60);

             auto it = mapCandidateInfo.find(time);
             //if (it == mapCandidateInfo.end())
             if (it != mapCandidateInfo.end())
             {
                if (allow_repeated_times)
                {
                   iterCurrentTime = it;
                   iCandidate = iterCurrentTime->second.size();
                   nCandidates += iCandidate;
                   iterCurrentTime->second.resize(iCandidate + nCandidates);

                }
                else
                {
                   TECA_ERROR("Repated time found in candidate files on line " << row)
                }
             }
             else
             {
                auto ins = mapCandidateInfo.insert(TimeToCandidateInfoPair(time,
                                    TimesliceCandidateInformation(nCandidates)));

                if (!ins.second)
                {
                   TECA_ERROR("Insertion of time "
                       << " into candidate info map failed on line " << row)
                   return -1;
                }

                iterCurrentTime = ins.first;
             }

             // Prepare to parse candidate data
             if (nCandidates != 0)
             {
                eReadState = ReadState_Candidate;
             }
          }

          if (eReadState == ReadState_Candidate)
          {
             TimesliceCandidateInformation &tscinfo = iterCurrentTime->second;

             for (size_t i = 0; i < vec_format_strings.size(); ++i)
             {
                if (vec_format_strings[i] == "lat")
                {
                   tscinfo[iCandidate].push_back(std::to_string(p_lat[row]));
                }
                else if (vec_format_strings[i] == "lon")
                {
                   tscinfo[iCandidate].push_back(std::to_string(p_lon[row]));
                }
                else if (vec_format_strings[i] == "i")
                {
                   tscinfo[iCandidate].push_back(std::to_string(p_ii[row]));
                }
                else if (vec_format_strings[i] == "j")
                {
                   tscinfo[iCandidate].push_back(std::to_string(p_jj[row]));
                }
                else
                {
                   auto var = candidates->get_column(vec_format_strings[i]);
                   auto [sp_var, p_var] = get_host_accessible<CTT>(var);
                   tscinfo[iCandidate].push_back(std::to_string(p_var[row]));
                }
             }

             // Advance candidate number
             iCandidate++;
             if (iCandidate == nCandidates)
             {
                eReadState = ReadState_Time;
                iterCurrentTime = mapCandidateInfo.end();
             }
          }
       }
    )

    return 0;
}

// --------------------------------------------------------------------------
struct Node
{
    double x;
    double y;
    double z;

    double lonRad;
    double latRad;
};

// --------------------------------------------------------------------------
/**
 * A (time,candidate index) pair
 */
struct TimeCandidatePair : public std::pair<int,int> {
	TimeCandidatePair(int _timeix, int _candidate) :
		std::pair<int,int>(_timeix, _candidate)
	{ }

	inline const int & timeix() const {
		return first;
	}

	inline const int & candidate() const {
		return second;
	}
};

// --------------------------------------------------------------------------
/**
 * A segment in a simple path connecting candidate indices at two times
 */
class SimplePathSegment :
	public std::pair<TimeCandidatePair, TimeCandidatePair>
{

public:
	//	Constructor.
	SimplePathSegment(
		int iTime0,
		int iCandidate0,
		int iTime1,
		int iCandidate1
	) :
		std::pair<TimeCandidatePair, TimeCandidatePair>(
			TimeCandidatePair(iTime0, iCandidate0),
			TimeCandidatePair(iTime1, iCandidate1))
	{ }

public:
	//	Comparator.
	bool operator< (const SimplePathSegment & seg) const {
		return (first < seg.first);
	}
};

typedef std::set<SimplePathSegment> SimplePathSegmentSet;

// --------------------------------------------------------------------------
/**
 * A vector of times and candidate indices at those times defining a path
 */
class SimplePath {

public:
	//	Array of times.
	std::vector<int> m_iTimes;

	//	Array of candidates.
	std::vector<int> m_iCandidates;
};

//	A vector of SimplePaths.
typedef std::vector<SimplePath> SimplePathVector;


// --------------------------------------------------------------------------
/**
 * An operator which is used to enforce some criteria on paths
 */
class PathThresholdOp {

public:
	//	Index denoting all points along path.
	static const int Count_All = (-1);

	//	Index denoting only the first point along the path.
	static const int Count_First = (-2);

	//	Index denoting only the last point along the path.
	static const int Count_Last = (-3);

	//	Possible operations.
	enum Operation {
		GreaterThan,
		LessThan,
		GreaterThanEqualTo,
		LessThanEqualTo,
		EqualTo,
		NotEqualTo,
		AbsGreaterThanEqualTo,
		AbsLessThanEqualTo
	};

public:
	//	Parse a threshold operator string.
	void Parse(
		const std::string &strOp,
		const std::vector<std::string> &vecFormatStrings
	) {
		// Read mode
		enum {
			ReadMode_Column,
			ReadMode_Op,
			ReadMode_Value,
			ReadMode_MinCount,
			ReadMode_Invalid
		} eReadMode = ReadMode_Column;

		// Loop through string
		int iLast = 0;
		for (long unsigned int i = 0; i <= strOp.length(); i++) {

			// Comma-delineated
			if ((i == strOp.length()) || (strOp[i] == ',')) {

				std::string strSubStr =
					strOp.substr(iLast, i - iLast);

				// Read in column name
				if (eReadMode == ReadMode_Column) {

					long unsigned int j = 0;
					for (; j < vecFormatStrings.size(); j++) {
						if (strSubStr == vecFormatStrings[j]) {
							m_iColumn = j;
							break;
						}
					}
					if (j == vecFormatStrings.size()) {
						TECA_ERROR("Threshold column name \"" << strSubStr << "\" "
							<< "not found in --format");
					}

					iLast = i + 1;
					eReadMode = ReadMode_Op;

				// Read in operation
				} else if (eReadMode == ReadMode_Op) {
					if (strSubStr == ">") {
						m_eOp = GreaterThan;
					} else if (strSubStr == "<") {
						m_eOp = LessThan;
					} else if (strSubStr == ">=") {
						m_eOp = GreaterThanEqualTo;
					} else if (strSubStr == "<=") {
						m_eOp = LessThanEqualTo;
					} else if (strSubStr == "=") {
						m_eOp = EqualTo;
					} else if (strSubStr == "!=") {
						m_eOp = NotEqualTo;
					} else if (strSubStr == "|>=") {
						m_eOp = AbsGreaterThanEqualTo;
					} else if (strSubStr == "|<=") {
						m_eOp = AbsLessThanEqualTo;
					} else {
						TECA_ERROR("Threshold invalid operation \"" << strSubStr << "\" ")
					}

					iLast = i + 1;
					eReadMode = ReadMode_Value;

				// Read in value
				} else if (eReadMode == ReadMode_Value) {
					m_dValue = atof(strSubStr.c_str());

					iLast = i + 1;
					eReadMode = ReadMode_MinCount;

				// Read in minimum count
				} else if (eReadMode == ReadMode_MinCount) {
					if (strSubStr == "all") {
						m_nMinimumCount = Count_All;
					} else if (strSubStr == "first") {
						m_nMinimumCount = Count_First;
					} else if (strSubStr == "last") {
						m_nMinimumCount = Count_Last;
					} else {
						m_nMinimumCount = atoi(strSubStr.c_str());
					}

					if (m_nMinimumCount < Count_Last) {
						TECA_ERROR("Invalid minimum count \""
							<< m_nMinimumCount << "\"");
					}

					iLast = i + 1;
					eReadMode = ReadMode_Invalid;

				// Invalid
				} else if (eReadMode == ReadMode_Invalid) {
					TECA_ERROR("Too many entries in threshold string \""
						<< strOp << "\"");
				}
			}
		}

		if (eReadMode != ReadMode_Invalid) {
			TECA_ERROR("Insufficient entries in threshold string \""
					<< strOp << "\"");
		}

		// Output announcement
		std::string strDescription;
		strDescription += vecFormatStrings[m_iColumn];
		if (m_eOp == GreaterThan) {
			strDescription += " greater than ";
		} else if (m_eOp == LessThan) {
			strDescription += " less than ";
		} else if (m_eOp == GreaterThanEqualTo) {
			strDescription += " greater than or equal to ";
		} else if (m_eOp == LessThanEqualTo) {
			strDescription += " less than or equal to ";
		} else if (m_eOp == EqualTo) {
			strDescription += " equal to ";
		} else if (m_eOp == NotEqualTo) {
			strDescription += " not equal to ";
		} else if (m_eOp == AbsGreaterThanEqualTo) {
			strDescription += " magnitude greater than or equal to ";
		} else if (m_eOp == AbsLessThanEqualTo) {
			strDescription += " magnitude less than or equal to ";
		}

		char szValue[128];
		snprintf(szValue, 128, "%f", m_dValue);
		strDescription += szValue;

		char szMinCount[160];
		if (m_nMinimumCount == Count_All) {
			strDescription += " at all times";
		} else if (m_nMinimumCount == Count_First) {
			strDescription += " at the first time";
		} else if (m_nMinimumCount == Count_Last) {
			strDescription += " at the last time";
		} else {
			snprintf(szMinCount, 160, " at least %i time(s)", m_nMinimumCount);
			strDescription += szMinCount;
		}

		Announce("%s", strDescription.c_str());
	}

	//	Verify that the specified path satisfies the threshold op.
	bool Apply(
		const SimplePath &path,
		const std::vector<TimeToCandidateInfoMapIterator> &vecCandidates
	) {
		int nCount = 0;
		for (long unsigned int s = 0; s < path.m_iTimes.size(); s++) {
			int t = path.m_iTimes[s];
			int i = path.m_iCandidates[s];

			if ((m_nMinimumCount == Count_First) && (s > 0)) {
				continue;
			}
			if ((m_nMinimumCount == Count_Last) && (s < path.m_iTimes.size()-1)) {
				continue;
			}

			_ASSERT((t >= 0) && (t < vecCandidates.size()));

			double dCandidateValue =
				std::stod(vecCandidates[t]->second[i][m_iColumn]);

			if ((m_eOp == GreaterThan) &&
				(dCandidateValue > m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == LessThan) &&
				(dCandidateValue < m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == GreaterThanEqualTo) &&
				(dCandidateValue >= m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == LessThanEqualTo) &&
				(dCandidateValue <= m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == EqualTo) &&
				(dCandidateValue == m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == NotEqualTo) &&
				(dCandidateValue != m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == AbsGreaterThanEqualTo) &&
				(fabs(dCandidateValue) >= m_dValue)
			) {
				nCount++;

			} else if (
				(m_eOp == AbsLessThanEqualTo) &&
				(fabs(dCandidateValue) <= m_dValue)
			) {
				nCount++;
			}
		}

		// Check that the criteria is satisfied for all segments
		if (m_nMinimumCount == Count_All) {
			if (nCount == (int)(path.m_iTimes.size())) {
				return true;
			} else {
				return false;
			}
		}

		// Check that the criteria are satisfied for the first or last segment
		if ((m_nMinimumCount == Count_First) || (m_nMinimumCount == Count_Last)) {
			if (nCount == 1) {
				return true;
			} else {
				return false;
			}
		}

		// Check total count against min count
		if (nCount >= m_nMinimumCount) {
			return true;
		} else {
			return false;
		}
	}

protected:
	//	Active column.
	int m_iColumn;

	//	Operation.
	Operation m_eOp;

	//	Threshold value.
	double m_dValue;

	//	Minimum number of segments that need to satisfy the op.
	int m_nMinimumCount;
};

// --------------------------------------------------------------------------
void GeneratePathSegmentsSetBasic(
	const std::vector<TimeToCandidateInfoMapIterator> &vecIterCandidates,
	const std::vector<std::vector<Node>> &vecNodes,
	const std::vector<kdtree*> &vecKDTrees,
	int nMaxGapSteps,
	double dMaxGapSeconds,
	double dRangeDeg,
	std::vector<SimplePathSegmentSet> & vecPathSegmentsSet
) {

	vecPathSegmentsSet.resize(vecIterCandidates.size()-1);

	// Null pointer
	int * noptr = NULL;

	// Search nodes at current time level
	for (size_t t = 0; t < vecIterCandidates.size()-1; t++) {

		const Time & timeCurrent = vecIterCandidates[t]->first;
		const TimesliceCandidateInformation & tscinfo = vecIterCandidates[t]->second;

		// Loop through all candidates at current time level
		for (long unsigned int i = 0; i < tscinfo.size(); i++) {

			// Check future timesteps for next candidate in path
			for (int g = 1; ; g++) {

				if ((nMaxGapSteps != (-1)) && (g > nMaxGapSteps+1)) {
					break;
				}
				if (t+g >= vecIterCandidates.size()) {
					break;
				}

				if (dMaxGapSeconds >= 0.0) {
					const Time & timeNext = vecIterCandidates[t+g]->first;
					double dDeltaSeconds = timeCurrent.DeltaSeconds(timeNext);
					if (dDeltaSeconds > dMaxGapSeconds) {
						break;
					}
				}

				if (vecKDTrees[t+g] == NULL) {
					continue;
				}

				kdres * set = kd_nearest3(vecKDTrees[t+g], vecNodes[t][i].x, vecNodes[t][i].y, vecNodes[t][i].z);

				if (kd_res_size(set) == 0) {
					kd_res_free(set);
					break;
				}

				int iRes =
					  reinterpret_cast<int*>(kd_res_item_data(set))
					- reinterpret_cast<int*>(noptr);

				kd_res_free(set);

				// Great circle distance between points
				double dRDeg =
					GreatCircleDistance_Deg(
						vecNodes[t][i].lonRad,
						vecNodes[t][i].latRad,
						vecNodes[t+g][iRes].lonRad,
						vecNodes[t+g][iRes].latRad);

				// Verify great circle distance satisfies range requirement
				if (dRDeg <= dRangeDeg) {

					// Insert new path segment into set of path segments
					vecPathSegmentsSet[t].insert(
						SimplePathSegment(t, i, t+g, iRes));

					break;
				}
			}
		}
	}
}

// --------------------------------------------------------------------------
/**
 * Priority method for generating path segments
 */
void GeneratePathSegmentsWithPriority(
	const std::vector<TimeToCandidateInfoMapIterator> &vecIterCandidates,
	const std::vector<std::vector<Node>> &vecNodes,
	const std::vector<kdtree*> &vecKDTrees,
	const int ixPriorityCol,
	int nMaxGapSteps,
	double dMaxGapSeconds,
	double dRangeDeg,
	std::vector<SimplePathSegmentSet> &vecPathSegmentsSet
) {
	// For target candidates, priority includes both timestep delta and distance
	typedef std::pair<int, double> PriorityPair;

	// Null pointer
	int * noptr = NULL;

	// Chord distance
	const double dChordLength = ChordLengthFromGreatCircleDistance_Deg(dRangeDeg);

	// Initialize the path segment set
	vecPathSegmentsSet.resize(vecIterCandidates.size()-1);

	// Get priority of all candidates
	if (ixPriorityCol < 0) {
		TECA_ERROR("Invalid priority column index");
	}
	std::multimap<double, TimeCandidatePair> mapPriority;
	for (size_t t = 0; t < vecIterCandidates.size()-1; t++) {
		const TimesliceCandidateInformation & tscinfo = vecIterCandidates[t]->second;

		for (long unsigned int i = 0; i < tscinfo.size(); i++) {
			if (tscinfo[i].size() <= ixPriorityCol) {
				TECA_ERROR("Priority column index out of range (" << tscinfo[i].size() << "<=" << ixPriorityCol << ")");
			}

			double dPriority = std::stod(tscinfo[i][ixPriorityCol]);
			mapPriority.insert(std::pair<double, TimeCandidatePair>(dPriority, TimeCandidatePair(t,i)));
		}
	}

	Announce("%lu total candidates found", mapPriority.size());

	// Set of candidates that are already targets of another node
	std::set<TimeCandidatePair> setUsedCandidates;

	// Loop through all candidates in increasing priority
	for (auto & it : mapPriority) {
		const long unsigned int t = it.second.timeix();
		const int i = it.second.candidate();

		const Time & timeCurrent = vecIterCandidates[t]->first;

		// Find all candidates within prescribed distance
		std::multimap<std::pair<int, double>, TimeCandidatePair> mmapTargets;
		for (int g = 1; ; g++) {

			if ((nMaxGapSteps != (-1)) && (g > nMaxGapSteps+1)) {
				break;
			}
			if (t+g >= vecKDTrees.size()) {
				break;
			}
			if (dMaxGapSeconds >= 0.0) {
				const Time & timeNext = vecIterCandidates[t+g]->first;
				double dDeltaSeconds = timeCurrent.DeltaSeconds(timeNext);
				if (dDeltaSeconds > dMaxGapSeconds) {
					break;
				}
			}
			if (vecKDTrees[t+g] == NULL) {
				continue;
			}

			kdres * set =
				kd_nearest_range3(
					vecKDTrees[t+g],
					vecNodes[t][i].x,
					vecNodes[t][i].y,
					vecNodes[t][i].z,
					dChordLength);

			if (set == NULL) {
				TECA_ERROR("Fatal exception in kd_nearest_range3");
			}
			if (kd_res_size(set) == 0) {
				kd_res_free(set);
				continue;
			}

			do {
				int iRes =
					  reinterpret_cast<int*>(kd_res_item_data(set))
					- reinterpret_cast<int*>(noptr);

				//printf("%i %i %i %i - %lu %lu\n", t, i, t+g, iRes);

				double dRDeg =
					GreatCircleDistance_Deg(
						vecNodes[t][i].lonRad,
						vecNodes[t][i].latRad,
						vecNodes[t+g][iRes].lonRad,
						vecNodes[t+g][iRes].latRad);

				mmapTargets.insert(
					std::pair<PriorityPair, TimeCandidatePair>(
						PriorityPair(g,dRDeg),
						TimeCandidatePair(t+g,iRes)));

			} while (kd_res_next(set));

			kd_res_free(set);
		}

		// Find the shortest target node and insert a segment
		for (auto & itTarget : mmapTargets) {
			auto itUsed = setUsedCandidates.find(itTarget.second);
			if (itUsed == setUsedCandidates.end()) {
				setUsedCandidates.insert(itTarget.second);

				vecPathSegmentsSet[t].insert(
					SimplePathSegment(t, i, itTarget.second.timeix(), itTarget.second.candidate()));

				break;
			}
		}
	}
}

// --------------------------------------------------------------------------
/**
 * First-come-first-serve method for generating paths (default).
 */
void GeneratePathsBasic(
	const std::vector<TimeToCandidateInfoMapIterator> & vecIterCandidates,
	std::vector<SimplePathSegmentSet> & vecPathSegmentsSet,
	std::vector<SimplePath> & vecPaths
) {
	// Loop through all times
	for (size_t t = 0; t < vecIterCandidates.size()-1; t++) {

		// Loop through all remaining segments at this time
		while (vecPathSegmentsSet[t].size() > 0) {

			// Create a new path starting with this segment
			SimplePath path;

			SimplePathSegmentSet::iterator iterSeg
				= vecPathSegmentsSet[t].begin();

			path.m_iTimes.push_back(iterSeg->first.timeix());
			path.m_iCandidates.push_back(iterSeg->first.candidate());

			int tx = t;

			for (;;) {
				path.m_iTimes.push_back(iterSeg->second.timeix());
				path.m_iCandidates.push_back(iterSeg->second.candidate());

				long unsigned int txnext = iterSeg->second.timeix();

				if (txnext >= vecIterCandidates.size()-1) {
					vecPathSegmentsSet[tx].erase(iterSeg);
					break;
				}

				SimplePathSegment segFind(
					iterSeg->second.timeix(), iterSeg->second.candidate(), 0, 0);

				vecPathSegmentsSet[tx].erase(iterSeg);

				iterSeg = vecPathSegmentsSet[txnext].find(segFind);

				if (iterSeg == vecPathSegmentsSet[txnext].end()) {
					break;
				}

				tx = txnext;
			}

			// Add path to array of paths
			vecPaths.push_back(path);
		}
	}
}

// --------------------------------------------------------------------------
// PIMPL idiom hides internals
class teca_stitch_nodes::internals_t
{
public:
    internals_t() {}
    ~internals_t() {}

    static
    int parse_variable_list(std::string &cmd,
                            std::vector<std::string> &vec);
    template<typename T>
    static
    int parse_threshold_list(std::vector<std::string> &vec_format_strings,
                             std::string &cmd,
                             std::vector<T> &vec);
public:
    std::vector<std::string> vec_format_strings;
    std::vector<PathThresholdOp> vec_threshold_op;

    double min_time_seconds;
    double max_gap_seconds;
    int max_gap_steps;
    long unsigned int min_time_steps;
    Time::CalendarType caltype;
};

// --------------------------------------------------------------------------
int teca_stitch_nodes::internals_t::parse_variable_list(
    std::string &cmd,
    std::vector<std::string> &vec)
{
    long unsigned int iVarBegin = 0;
    long unsigned int iVarCurrent = 0;

    // Parse variable name
    for (;;)
    {
       if ((iVarCurrent >= cmd.length()) ||
       	   (cmd[iVarCurrent] == ',') ||
       	   (cmd[iVarCurrent] == ' ') ||
       	   (cmd[iVarCurrent] == '\t') ||
       	   (cmd[iVarCurrent] == '\n') ||
       	   (cmd[iVarCurrent] == '\r'))
       {
          if (iVarCurrent == iVarBegin)
          {
             if (iVarCurrent >= cmd.length())
             {
                break;
             }

             iVarCurrent++;
             iVarBegin++;
             continue;
          }
          vec.push_back(cmd.substr(iVarBegin, iVarCurrent - iVarBegin));
          iVarBegin = iVarCurrent + 1;
       }

       iVarCurrent++;
    }
    return 0;
}

// --------------------------------------------------------------------------
template<typename T>
int teca_stitch_nodes::internals_t::parse_threshold_list(
    std::vector<std::string> &vec_format_strings,
    std::string &cmd,
    std::vector<T> &vec)
{
    int i_last = 0;
    for (long unsigned int i = 0; i <= cmd.length(); ++i)
    {
       if ((i == cmd.length()) ||
           (cmd[i] == ';') ||
           (cmd[i] == ':'))
       {
          std::string strSubStr = cmd.substr(i_last, i - i_last);

          int i_next_op = (int)(vec.size());
          vec.resize(i_next_op + 1);
          vec[i_next_op].Parse(strSubStr, vec_format_strings);

          i_last = i + 1;
       }
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_stitch_nodes::teca_stitch_nodes() :
    in_connect(""),
    in_fmt(""),
    min_time("10"),
    cal_type("standard"),
    max_gap("3"),
    threshold(""),
    prioritize(""),
    min_path_length(1),
    range(8.0),
    min_endpoint_distance(0.0),
    min_path_distance(0.0),
    allow_repeated_times(false)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->internals = new teca_stitch_nodes::internals_t;
}

// --------------------------------------------------------------------------
teca_stitch_nodes::~teca_stitch_nodes()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_stitch_nodes::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_stitch_nodes":prefix));

    //TODO
    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, in_connect, "")
        TECA_POPTS_GET(std::string, prefix, in_fmt, "")
        TECA_POPTS_GET(std::string, prefix, min_time, "")
        TECA_POPTS_GET(std::string, prefix, cal_type, "")
        TECA_POPTS_GET(std::string, prefix, max_gap, "")
        TECA_POPTS_GET(std::string, prefix, threshold, "")
        TECA_POPTS_GET(std::string, prefix, prioritize, "")
        TECA_POPTS_GET(int, prefix, min_path_length, "")
        TECA_POPTS_GET(double, prefix, range, "")
        TECA_POPTS_GET(double, prefix, min_endpoint_distance, "")
        TECA_POPTS_GET(double, prefix, min_path_distance, "")
        TECA_POPTS_GET(bool, prefix, allow_repeated_times, "")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_stitch_nodes::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, in_connect)
    TECA_POPTS_SET(opts, std::string, prefix, in_fmt)
    TECA_POPTS_SET(opts, std::string, prefix, min_time)
    TECA_POPTS_SET(opts, std::string, prefix, cal_type)
    TECA_POPTS_SET(opts, std::string, prefix, max_gap)
    TECA_POPTS_SET(opts, std::string, prefix, threshold)
    TECA_POPTS_SET(opts, std::string, prefix, prioritize)
    TECA_POPTS_SET(opts, int, prefix, min_path_length)
    TECA_POPTS_SET(opts, double, prefix, range)
    TECA_POPTS_SET(opts, double, prefix, min_endpoint_distance)
    TECA_POPTS_SET(opts, double, prefix, min_path_distance)
    TECA_POPTS_SET(opts, bool, prefix, allow_repeated_times)
}
#endif

// --------------------------------------------------------------------------
int teca_stitch_nodes::initialize()
{
    // Parse minimum time
    this->internals->min_time_steps = this->min_path_length;
    this->internals->min_time_seconds = 0.0;
    if (STLStringHelper::IsIntegerIndex(this->min_time))
    {
       this->internals->min_time_steps = std::stoi(this->min_time);
       if (this->internals->min_time_steps < 1)
       {
    	    TECA_ERROR("Invalid value of --min_time;"
    		   " expected positive integer");
          return -1;
       }

    }
    else
    {
       Time timeMinTime;
       timeMinTime.FromFormattedString(this->min_time);
       this->internals->min_time_seconds = timeMinTime.AsSeconds();
       if (this->internals->min_time_seconds <= 0.0)
       {
          TECA_ERROR("Invalid value of --min_time;"
              " expected positive integer");
          return -1;
       }
    }

    // Parse maximum gap
    this->internals->max_gap_steps = (-1);
    this->internals->max_gap_seconds = -1.0;
    if (STLStringHelper::IsIntegerIndex(this->max_gap))
    {
       this->internals->max_gap_steps = std::stoi(this->max_gap);
       if (this->internals->max_gap_steps < 0)
       {
    	    TECA_ERROR("Invalid value of --max_gap;"
    		   " expected nonnegative integer");
          return -1;
       }
    }
    else
    {
       Time timeMaxGap;
       timeMaxGap.FromFormattedString(this->max_gap);
       this->internals->max_gap_seconds = timeMaxGap.AsSeconds();
       if (this->internals->max_gap_seconds < 0.0)
       {
          TECA_ERROR("Invalid value of --max_gap;"
              " expected nonnegative integer");
          return -1;
       }
    }

    // Parse calendar type
    this->internals->caltype = Time::CalendarTypeFromString(this->cal_type);

    if (this->internals->caltype == Time::CalendarNone)
    {
       if (this->internals->min_time_seconds > 0.0)
       {
     	    TECA_ERROR("A calendar type (--caltype) must be specified");
          return -1;
       }
       if (this->internals->max_gap_seconds >= 0.0)
       {
     	    TECA_ERROR("A calendar type (--caltype) must be specified");
          return -1;
       }
    }

    // Check for connectivity file
    if (this->in_connect != "")
    {
       TECA_ERROR("Loading grid data from connectivity file"
           "is not supported")
       return -1;
    }

    // Parse format string
    if (internals_t::parse_variable_list(this->in_fmt,
        this->internals->vec_format_strings))
       return -1;

    if (this->internals->vec_format_strings.size() == 0)
    {
    	 TECA_ERROR("No format specified");
       return -1;
    }

    // Parse the threshold string
    if (this->threshold != "")
    {
       if (internals_t::parse_threshold_list(this->internals->vec_format_strings,
           this->threshold,
           this->internals->vec_threshold_op))
          return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_stitch_nodes::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_stitch_nodes::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    teca_metadata req(request);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_stitch_nodes::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_stitch_nodes:execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_table candidates =
        std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif
    if (!candidates)
    {
        if (rank == 0)
        {
            TECA_FATAL_ERROR("empty input or not a table")
        }
        return nullptr;
    }

    // check that there are some candidates to work with.
    unsigned long n_rows = candidates->get_number_of_rows();
    if (n_rows < 1)
    {
        TECA_FATAL_ERROR("Failed to form TC tracks because there were no candidates")
        return nullptr;
    }

    // Check format string for lat/lon
    int iLatIndex = (-1);
    int iLonIndex = (-1);

    // validate the input table
    for (size_t i = 0; i < this->internals->vec_format_strings.size(); ++i)
    {
       if (!candidates->has_column(this->internals->vec_format_strings[i]))
       {
           TECA_FATAL_ERROR("Candidate table missing \""
               << this->internals->vec_format_strings[i] << "\"")
           return nullptr;
       }

       if (this->internals->vec_format_strings[i] == "lat")
       {
          iLatIndex = i;
       }
       if (this->internals->vec_format_strings[i] == "lon")
       {
          iLonIndex = i;
       }
    }

    if (iLatIndex == (-1))
    {
       TECA_ERROR("Latitude \"lat\" must be specified in format (--in_fmt)");
       return nullptr;
    }
    if (iLonIndex == (-1))
    {
       TECA_ERROR("Latitude \"lon\" must be specified in format (--in_fmt)");
       return nullptr;
    }

    // Parse the input into mapCandidateInfo, then store iterators to all elements
    // of mapCandidateInfo in vecIterCandidates to enable sequential access.
    TimeToCandidateInfoMap mapCandidateInfo;
    std::vector<TimeToCandidateInfoMapIterator> vecIterCandidates;

    AnnounceStartBlock("Loading candidate data");

    if (parse_detect_nodes_file(candidates, this->internals->vec_format_strings,
                                mapCandidateInfo, this->internals->caltype,
                                this->allow_repeated_times))
    {
       TECA_FATAL_ERROR("parse_detect_nodes_file encountered an error")
       return nullptr;
    }

    vecIterCandidates.reserve(mapCandidateInfo.size());
    for (auto it = mapCandidateInfo.begin(); it != mapCandidateInfo.end(); it++)
    {
       vecIterCandidates.push_back(it);
    }

    const Time &timeFirst = vecIterCandidates[0]->first;
    const Time &timeLast = vecIterCandidates[vecIterCandidates.size()-1]->first;

    // Verify total time is larger than mintime
    if (this->internals->min_time_seconds > 0.0)
    {
       double delta_seconds = timeFirst.DeltaSeconds(timeLast);
       if (this->internals->min_time_seconds > delta_seconds)
       {
          TECA_FATAL_ERROR("Duration of record "
              << "is shorter than --min_time; no paths possible")
          return nullptr;
       }
    }

    // Verify adjacent times are smaller than maxgap
    if (this->internals->max_gap_seconds >= 0.0)
    {
       for (size_t t = 0; t < vecIterCandidates.size()-1; t++)
       {
          const Time &timeCurrent = vecIterCandidates[t]->first;
          const Time &timeNext = vecIterCandidates[t+1]->first;

          double delta_seconds = timeCurrent.DeltaSeconds(timeNext);
          if (delta_seconds > this->internals->max_gap_seconds)
          {
             TECA_FATAL_ERROR("Discrete times " << t << " and " << t+1
                 << "differ by more than --max_gap")
             return nullptr;
          }
       }
    }
    AnnounceEndBlock("Done");

    AnnounceStartBlock("Creating KD trees at each time level");

    // Null pointer
    int * noptr = NULL;

    // Vector of lat/lon values
    std::vector< std::vector<Node> > vecNodes;
    vecNodes.resize(mapCandidateInfo.size());

    // Vector of KD trees
    std::vector<kdtree *> vecKDTrees;
    vecKDTrees.resize(mapCandidateInfo.size());

    for (size_t t = 0; t < vecIterCandidates.size(); t++)
    {
       const TimesliceCandidateInformation & tscinfo = vecIterCandidates[t]->second;

       // Create a new kdtree
       if (tscinfo.size() == 0)
       {
          vecKDTrees[t] = NULL;
          continue;
       }

       vecKDTrees[t] = kd_create(3);

       vecNodes[t].resize(tscinfo.size());

       // Insert all points at this time level
       for (long unsigned int i = 0; i < tscinfo.size(); i++)
       {
          vecNodes[t][i].lonRad = DegToRad(std::stod(tscinfo[i][iLonIndex]));
          vecNodes[t][i].latRad = DegToRad(std::stod(tscinfo[i][iLatIndex]));

          RLLtoXYZ_Rad(vecNodes[t][i].lonRad, vecNodes[t][i].latRad,
                       vecNodes[t][i].x, vecNodes[t][i].y, vecNodes[t][i].z);

          kd_insert3(vecKDTrees[t], vecNodes[t][i].x, vecNodes[t][i].y,
                     vecNodes[t][i].z, reinterpret_cast<void*>(noptr+i));
       }
    }
    AnnounceEndBlock("Done");

    AnnounceStartBlock("Constructing paths");

    std::vector<SimplePathSegmentSet> vecPathSegmentsSet;

    AnnounceStartBlock("Populating set of path segments");

    // Vector over time of sets of all path segments that start at that time
    if (this->prioritize == "")
    {
       GeneratePathSegmentsSetBasic(vecIterCandidates,
                                    vecNodes,
                                    vecKDTrees,
                                    this->internals->max_gap_steps,
                                    this->internals->max_gap_seconds,
                                    this->range,
                                    vecPathSegmentsSet);
    }
    else
    {
       int ixPriorityCol = (-1);
       for (long unsigned int i = 0; i < this->internals->vec_format_strings.size(); i++)
       {
          if (this->internals->vec_format_strings[i] == this->prioritize)
          {
             ixPriorityCol = i;
             break;
          }
       }

       if (ixPriorityCol == (-1))
       {
          TECA_ERROR("Format string (--in_fmt) does not contain priority column \"" << this->prioritize << "\"")
       }

       GeneratePathSegmentsWithPriority(vecIterCandidates,
                                        vecNodes,
                                        vecKDTrees,
                                        ixPriorityCol,
                                        this->internals->max_gap_steps,
                                        this->internals->max_gap_seconds,
                                        this->range,
                                        vecPathSegmentsSet);
    }

    AnnounceEndBlock("Done");

    AnnounceStartBlock("Connecting path segments");

    std::vector<SimplePath> vecPaths;

    GeneratePathsBasic(vecIterCandidates,
                       vecPathSegmentsSet,
                       vecPaths);

    AnnounceEndBlock("Done");

    AnnounceEndBlock("Done");

    AnnounceStartBlock("Filtering paths");

    int nRejectedMinTimePaths = 0;
    int nRejectedMinEndpointDistPaths = 0;
    int nRejectedMinPathDistPaths = 0;
    int nRejectedThresholdPaths = 0;

    std::vector<SimplePath> vecPathsUnfiltered = vecPaths;
    vecPaths.clear();

    // Loop through all paths
    for (const SimplePath & path : vecPathsUnfiltered)
    {
       // Reject path due to minimum time
       if (path.m_iTimes.size() < this->internals->min_time_steps)
       {
          nRejectedMinTimePaths++;
          continue;
       }
       if (this->internals->min_time_steps > 0.0)
       {
          int nT = path.m_iTimes.size();
          int iTime0 = path.m_iTimes[0];
          int iTime1 = path.m_iTimes[nT-1];

          const Time &timeFirst = vecIterCandidates[iTime0]->first;
          const Time &timeLast = vecIterCandidates[iTime1]->first;

          double dDeltaSeconds = timeFirst.DeltaSeconds(timeLast);
          if (dDeltaSeconds < this->internals->min_time_seconds)
          {
             nRejectedMinTimePaths++;
             continue;
          }
       }

       // Reject path due to minimum endpoint distance
       if (this->min_endpoint_distance > 0.0)
       {
          int nT = path.m_iTimes.size();

          int iTime0 = path.m_iTimes[0];
          int iRes0  = path.m_iCandidates[0];

          int iTime1 = path.m_iTimes[nT-1];
          int iRes1  = path.m_iCandidates[nT-1];

          double dLonRad0 = vecNodes[iTime0][iRes0].lonRad;
          double dLatRad0 = vecNodes[iTime0][iRes0].latRad;

          double dLonRad1 = vecNodes[iTime1][iRes1].lonRad;
          double dLatRad1 = vecNodes[iTime1][iRes1].latRad;

          double dR = GreatCircleDistance_Deg(dLonRad0, dLatRad0, dLonRad1, dLatRad1);

          if (dR < this->min_endpoint_distance)
          {
             nRejectedMinEndpointDistPaths++;
             continue;
          }
       }

       // Reject path due to minimum total path distance
       if (this->min_path_distance > 0.0)
       {
          double dTotalPathDistance = 0.0;
          for (long unsigned int i = 0; i < path.m_iTimes.size() - 1; i++)
          {
             int iTime0 = path.m_iTimes[i];
             int iRes0 = path.m_iCandidates[i];

             int iTime1 = path.m_iTimes[i+1];
             int iRes1 = path.m_iCandidates[i+1];

             double dLonRad0 = vecNodes[iTime0][iRes0].lonRad;
             double dLatRad0 = vecNodes[iTime0][iRes0].latRad;

             double dLonRad1 = vecNodes[iTime1][iRes1].lonRad;
             double dLatRad1 = vecNodes[iTime1][iRes1].latRad;

             double dR = GreatCircleDistance_Deg(dLonRad0, dLatRad0, dLonRad1, dLatRad1);

             dTotalPathDistance += dR;
          }

          if (dTotalPathDistance < this->min_path_distance)
          {
             nRejectedMinPathDistPaths++;
             continue;
          }
       }

       // Reject path due to threshold
       bool fOpResult = true;
       for (long unsigned int x = 0; x < this->internals->vec_threshold_op.size(); x++)
       {
          fOpResult = this->internals->vec_threshold_op[x].Apply(path, vecIterCandidates);

          if (!fOpResult)
          {
             break;
          }
       }
       if (!fOpResult)
       {
          nRejectedThresholdPaths++;
          continue;
       }

       // Add path to array of paths
       vecPaths.push_back(path);
    }

    Announce("Paths rejected (mintime): %i", nRejectedMinTimePaths);
    Announce("Paths rejected (minendpointdist): %i", nRejectedMinEndpointDistPaths);
    Announce("Paths rejected (minpathdist): %i", nRejectedMinPathDistPaths);
    Announce("Paths rejected (threshold): %i", nRejectedThresholdPaths);
    Announce("Total paths found: %i", vecPaths.size());
    AnnounceEndBlock("Done");

    AnnounceStartBlock("Writing results");

    // create the table to hold storm tracks
    p_teca_table storm_tracks = teca_table::New();
    storm_tracks->copy_metadata(candidates);

    std::string time_units;
    storm_tracks->get_time_units(time_units);

    storm_tracks->declare_columns("storm_id", long(), "path_length", long(),
                                  "year", int(), "month", int(), "day", int(), "hour", int());

    // Copy over column data from candidate file to track file for output
    n_rows = 0;
    for (long unsigned int i = 0; i < vecPaths.size(); i++)
    {
       n_rows += vecPaths[i].m_iTimes.size();
    }


    for (long unsigned int var = 0; var < this->internals->vec_format_strings.size(); var++)
    {
       int row = 0;
       double *d_output_array = new double[n_rows];
       int    *i_output_array = new int[n_rows];
       for (long unsigned int i = 0; i < vecPaths.size(); i++)
       {
          for (long unsigned int t = 0; t < vecPaths[i].m_iTimes.size(); t++)
          {
             int iTime = vecPaths[i].m_iTimes[t];

             int iCandidate = vecPaths[i].m_iCandidates[t];

             if (var == 0)
             {
                storm_tracks << i << vecPaths[i].m_iTimes.size()
                                  << vecIterCandidates[iTime]->first.GetYear()
                                  << vecIterCandidates[iTime]->first.GetMonth()
                                  << vecIterCandidates[iTime]->first.GetDay()
                                  << ceil(vecIterCandidates[iTime]->first.GetSecond() / 3600.);
             }

             if (this->internals->vec_format_strings[var] == "i" ||
                 this->internals->vec_format_strings[var] == "j")
                i_output_array[row] = std::stoi(vecIterCandidates[iTime]->second[iCandidate][var]);
             else
                d_output_array[row] = std::stod(vecIterCandidates[iTime]->second[iCandidate][var]);
             row++;
          }
       }


       if (this->internals->vec_format_strings[var] == "i" ||
           this->internals->vec_format_strings[var] == "j")
       {
          p_teca_variant_array_impl<int> output =
             teca_variant_array_impl<int>::New(n_rows, i_output_array);
          storm_tracks->append_column(this->internals->vec_format_strings[var], output);
       }
       else
       {
          p_teca_variant_array_impl<double> output =
             teca_variant_array_impl<double>::New(n_rows, d_output_array);
          storm_tracks->append_column(this->internals->vec_format_strings[var], output);
       }

       delete [] d_output_array;
       delete [] i_output_array;
    }

    AnnounceEndBlock("Done");

    // Cleanup
    AnnounceStartBlock("Cleanup");

    for (long unsigned int t = 0; t < vecKDTrees.size(); t++)
    {
       kd_free(vecKDTrees[t]);
    }

    AnnounceEndBlock("Done");

    return storm_tracks;
}
