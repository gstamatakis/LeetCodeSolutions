import unittest

from python.LeetCode import *


class LeetTests(unittest.TestCase):
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
        a1 = ListNode(2)
        a12 = ListNode(4)
        a13 = ListNode(3)
        a1.next = a12
        a12.next = a13

        b1 = ListNode(5)
        b12 = ListNode(6)
        b13 = ListNode(4)
        b1.next = b12
        b12.next = b13

        c1 = ListNode(7)
        c12 = ListNode(0)
        c13 = ListNode(8)
        c1.next = c12
        c12.next = c13

        res = addTwoNumbers(a1, b1)

        while c1 or res:
            self.assertEqual(c1.val, res.val)
            c1 = (c1.next if c1 else None)
            res = (res.next if res else None)

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
        self.assertEqual(longest_common_prefix(["flower", "flow", "flight"]), "fl")
        self.assertEqual(longest_common_prefix(["dog", "racecar", "car"]), "")

    # 15
    def test_three_sum(self):
        self.assertEqual(threeSum([-1, 0, 1, 2, -1, -4]), [[-1, 1, 0], [-1, 2, -1]])
        self.assertEqual(threeSum([1, 1, 1, 1, 1]), [])
        self.assertEqual(threeSum([0, 0, 0]), [[0, 0, 0]])
        self.assertEqual(threeSum([0, 0, 0, 0, 0]), [[0, 0, 0]])

    def test_letter_combinations(self):
        self.assertEqual(letter_combinations("23"), ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])

    # 739
    def test_daily_temperature(self):
        self.assertEqual(daily_temperature([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])


if __name__ == '__main__':
    unittest.main()