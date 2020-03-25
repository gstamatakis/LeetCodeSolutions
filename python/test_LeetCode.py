import unittest

from python.LeetCode import *


class LeetTests(unittest.TestCase):
    # UTILS
    def listOfListsEqual(self, l1, l2):
        if not l1 or l2:
            return False
        return sorted(l1) == sorted(l2)

    # 1
    def test_twoSum(self):
        """
        Given nums = [2, 7, 11, 15], target = 9,
        Because nums[0] + nums[1] = 2 + 7 = 9,
        return [0, 1].
        """
        self.assertEqual(twoSum([2, 7, 11, 15], 9), [0, 1])

    # 2
    def test_addTwoNumbers(self):
        """
        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 0 -> 8
        Explanation: 342 + 465 = 807.
        """
        a, b, c = ListNode([2, 4, 3]), ListNode([5, 6, 4]), ListNode([7, 0, 8])
        res = addTwoNumbers(a, b)
        self.assertListEqual(res.toList(), c.toList())

    # 3
    def test_lengthOfLongestSubstring(self):
        """
        Input: "pwwkew"
        Output: 3
        Explanation: The answer is "wke", with the length of 3.
                     Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
        """
        self.assertEqual(lengthOfLongestSubstring("pwwkew"), 3)

    # 4
    def test_findMedianSortedArrays(self):
        self.assertEqual(findMedianSortedArrays([1, 3], [2]), 2)

    # 5
    def test_longestPalindrome(self):
        # Some strings accepts multiple solutions
        self.assertEqual(longestPalindrome("cbbd"), "bb")
        self.assertTrue(longestPalindrome("babad") in ["bab", "aba"])
        self.assertEqual(longestPalindromeDP("cbbd"), "bb")
        self.assertTrue(longestPalindromeDP("babad") in ["bab", "aba"])

    # 6
    def test_zig_zag(self):
        """
        Input: s = "PAYPALISHIRING", numRows = 4
        Output: "PINALSIGYAHRPI"
        Explanation:

        P     I    N
        A   L S  I G
        Y A   H R
        P     I
        """
        res = zig_zag_conversion("PAYPALISHIRING", 4)
        self.assertEqual(res, "PINALSIGYAHRPI")

    # 7
    def test_reverse_integer(self):
        self.assertEqual(reverse_integer(-123), -321)

    # 8
    def test_my_atoi(self):
        """
        Input: "   -42"
        Output: -42
        Explanation: The first non-whitespace character is '-', which is the minus sign.
                     Then take as many numerical digits as possible, which gets 42.
        """
        self.assertEqual(myAtoi("1234"), 1234)
        self.assertEqual(myAtoi("   -42"), -42)
        self.assertEqual(myAtoi(" "), 0)

    # 9
    def test_palindrome_number(self):
        self.assertTrue(palindrome_number(121))
        self.assertFalse(palindrome_number(-123))
        self.assertTrue(palindrome_number(11))

    # 10
    def test_regex_match(self):
        self.assertFalse(regex_match("aa", "a"))
        self.assertTrue(regex_match("aa", "a*"))
        self.assertTrue(regex_match("ab", ".*"))
        self.assertTrue(regex_match("aab", "c*a*b"))
        self.assertFalse(regex_match("mississippi", "mis*is*p*."))

    # 11
    def test_container_most_water(self):
        self.assertEqual(container_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]), 49)

    # 12
    def test_int_to_roman(self):
        self.assertEqual(integer_to_roman(1994), "MCMXCIV")

    # 13
    def test_roman_to_int(self):
        self.assertEqual(roman_to_integer("MCMXCIV"), 1994)

    # 14
    def test_longest_common_prefix(self):
        self.assertEqual(longest_common_prefix(
            ["flower", "flow", "flight"]), "fl")
        self.assertEqual(longest_common_prefix(["dog", "racecar", "car"]), "")

    # 15
    def test_three_sum(self):
        self.assertEqual(
            threeSum([-1, 0, 1, 2, -1, -4]), [[-1, 1, 0], [-1, 2, -1]])
        self.assertEqual(threeSum([1, 1, 1, 1, 1]), [])
        self.assertEqual(threeSum([0, 0, 0]), [[0, 0, 0]])
        self.assertEqual(threeSum([0, 0, 0, 0, 0]), [[0, 0, 0]])

    # 19
    def test_remove_nth_from_end(self):
        ll1 = ListNode([1, 2, 3, 4, 5])
        ll2 = ListNode([1, 2, 3, 5])
        res = removeNthFromEnd(ll1, 2)
        self.assertListEqual(ll2.toList(), res.toList())

    # 20
    def test_valid_parenthesis(self):
        self.assertTrue(validParenthesis('()'))
        self.assertTrue(validParenthesis('()[]{}'))
        self.assertFalse(validParenthesis('(]'))
        self.assertFalse(validParenthesis('([)]'))
        self.assertTrue(validParenthesis('{[]}'))

    # 21
    def test_merge_two_sorted_lists(self):
        ll1 = ListNode([1, 2, 4])
        ll2 = ListNode([1, 3, 4])
        ll3 = ListNode([1, 1, 2, 3, 4, 4])
        self.assertListEqual(mergeTwoLists(ll1, ll2).toList(), ll3.toList())

    # 22
    def test_generate_parenthesis(self):
        self.assertListEqual(generateParenthesis(0), [''])
        self.assertListEqual(generateParenthesis(3), ["((()))",
                                                      "(()())",
                                                      "(())()",
                                                      "()(())",
                                                      "()()()"])

    # 23
    def test_merge_k_sorted_lists(self):
        """
        1->4->5,
        1->3->4,
        2->6
        ]
        Output: 1->1->2->3->4->4->5->6
        """
        ll1 = ListNode([1, 4, 5])
        ll2 = ListNode([1, 3, 4])
        ll3 = ListNode([2, 6])
        res = ListNode([1, 1, 2, 3, 4, 4, 5, 6])
        self.assertEqual(mergeKLists([ll1, ll2, ll3]).toList(), res.toList())

    # 26
    def test_remove_duplicates_sorted_array(self):
        arr = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        self.assertEqual(removeDuplicates(arr), 5)

    # 28
    def test_strstr(self):
        haystack = "hello"
        needle = "ll"
        self.assertEqual(strStr(haystack, needle), 2)

        haystack = "aaaaa"
        needle = "bba"
        self.assertEqual(strStr(haystack, needle), -1)

        haystack = "a"
        needle = "a"
        self.assertEqual(strStr(haystack, needle), 0)

    # 29
    def test_divide_two_integers(self):
        dividend, divisor = 10, 3
        self.assertEqual(divide(dividend, divisor), 3)

    # 33
    def test_search_rotated_sorted_array(self):
        nums = [4, 5, 6, 7, 0, 1, 2]
        target = 0
        self.assertEqual(search(nums, target), 4)

    # 42
    def test_trap(self):
        height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
        target = 6
        self.assertEqual(trap_brute_force(height), target)
        self.assertEqual(trap_DP(height), target)
        self.assertEqual(trap_2P(height), target)
        self.assertEqual(trap_Stack(height), target)

    # 44
    def test_pattern_matching(self):
        """
        Example 1:
        Input:
        s = "aa"
        p = "a"
        Output: false
        Explanation: "a" does not match the entire string "aa".
        Example 2:

        Input:
        s = "aa"
        p = "*"
        Output: true
        Explanation: '*' matches any sequence.
        Example 3:

        Input:
        s = "cb"
        p = "?a"
        Output: false
        Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
        Example 4:

        Input:
        s = "adceb"
        p = "*a*b"
        Output: true
        Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
        Example 5:

        Input:
        s = "acdcb"
        p = "a*c?b"
        Output: false
        """
        self.assertEqual(wildcard_match('aa', 'a'), False)
        self.assertEqual(wildcard_match('aa', '*'), True)
        self.assertEqual(wildcard_match('cb', '?a'), False)
        self.assertEqual(wildcard_match('adceb', '*a*b'), True)
        self.assertEqual(wildcard_match('acdcb', 'a*c?b'), False)

        self.assertEqual(wildcard_match_DP('aa', 'a'), False)
        self.assertEqual(wildcard_match_DP('aa', '*'), True)
        self.assertEqual(wildcard_match_DP('cb', '?a'), False)
        self.assertEqual(wildcard_match_DP('adceb', '*a*b'), True)
        self.assertEqual(wildcard_match_DP('acdcb', 'a*c?b'), False)

    # 46
    def test_permute(self):
        input = [1, 2, 3]
        expected = [
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [2, 3, 1],
            [3, 1, 2],
            [3, 2, 1]
        ]
        self.listOfListsEqual(permute_dfs(input), expected)
        self.listOfListsEqual(permute_reduce(input), expected)

    # 48
    def test_rotate(self):
        """
        Example 1:
        Given input matrix =
        [
          [1,2,3],
          [4,5,6],
          [7,8,9]
        ],

        rotate the input matrix in-place such that it becomes:
        [
          [7,4,1],
          [8,5,2],
          [9,6,3]
        ]
        Example 2:

        Given input matrix =
        [
          [ 5, 1, 9,11],
          [ 2, 4, 8,10],
          [13, 3, 6, 7],
          [15,14,12,16]
        ],

        rotate the input matrix in-place such that it becomes:
        [
          [15,13, 2, 5],
          [14, 3, 4, 1],
          [12, 6, 8, 9],
          [16, 7,10,11]
        ]
        """
        matrix = [
            [5, 1, 9, 11],
            [2, 4, 8, 10],
            [13, 3, 6, 7],
            [15, 14, 12, 16]
        ]
        rotate(matrix)
        expected = [[15, 13, 2, 5],
                    [14, 3, 4, 1],
                    [12, 6, 8, 9],
                    [16, 7, 10, 11]]

        self.listOfListsEqual(matrix, expected)

        matrix = [
            [5, 1, 9, 11],
            [2, 4, 8, 10],
            [13, 3, 6, 7],
            [15, 14, 12, 16]
        ]
        rotate_B(matrix)
        self.listOfListsEqual(matrix, expected)

    # 49
    def test_groupAnagrams(self):
        input = ["eat", "tea", "tan", "ate", "nat", "bat"]
        expected = [
            ["ate", "eat", "tea"],
            ["nat", "tan"],
            ["bat"]
        ]
        self.listOfListsEqual(groupAnagrams(input), expected)

    # 50
    def test_myPow(self):
        self.assertEqual(myPow(2, 10.0), 1024.0)

    # 53
    def test_subarraySum(self):
        target = 6
        input = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        self.assertEqual(maxSubArray(input), target)
        self.assertEqual(maxSubArrayDnC(input), target)

    def test_letter_combinations(self):
        self.assertEqual(letter_combinations("23"), [
            "ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])

    # 739
    def test_daily_temperature(self):
        self.assertEqual(daily_temperature(
            [73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])


if __name__ == '__main__':
    unittest.main()
