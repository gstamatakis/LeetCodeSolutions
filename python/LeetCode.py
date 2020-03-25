###UTILS###
# Definition for singly-linked list.
from functools import reduce
from typing import List

from math import floor


class ListNode:
    def __init__(self, x):
        if isinstance(x, list):
            self.val = x[0]
            self.next = None
            for val in x[1:]:
                self.insert(val)
        else:
            self.val = x
            self.next = None

    def insert(self, x):
        last = self
        while last.next:
            last = last.next
        last.next = ListNode(x)

    def toList(self):
        res = []
        cur = self
        while cur:
            res.append(cur.val)
            cur = cur.next
        return res


###SOLUTIONS###

# 1
def twoSum(nums, target):
    """
    Given an array of integers, return indices of the two numbers such that they add up to a specific target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Runtime: 60 ms, faster than 70.19% of Python3 online submissions for Two Sum.
    Memory Usage: 14.9 MB, less than 10.46% of Python3 online submissions for Two Sum.
    """
    h = {}
    for i, num in enumerate(nums):
        n = target - num
        if n not in h:
            h[num] = i
        else:
            return [h[n], i]


# No 2
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    """
    You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse
    order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    """
    result = ListNode(0)
    result_tail = result
    carry = 0

    while l1 or l2 or carry:
        val1 = (l1.val if l1 else 0)
        val2 = (l2.val if l2 else 0)
        carry, out = divmod(val1 + val2 + carry, 10)

        result_tail.next = ListNode(out)
        result_tail = result_tail.next

        l1 = (l1.next if l1 else None)
        l2 = (l2.next if l2 else None)

    return result.next


# 3
def lengthOfLongestSubstring(s: str) -> int:
    """
    Given a string, find the length of the longest substring without repeating characters.
    """
    dicts = {}
    maxlength = start = 0
    for i, value in enumerate(s):
        if value in dicts:
            sums = dicts[value] + 1
            if sums > start:
                start = sums
        num = i - start + 1
        if num > maxlength:
            maxlength = num
        dicts[value] = i
    return maxlength


# No 4
def findMedianSortedArrays(num1, num2):
    """
    There are two sorted arrays nums1 and nums2 of size m and n respectively.
    Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
    You may assume nums1 and nums2 cannot be both empty.

    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    m, n = len(num1), len(num2)
    if m > n:
        num1, num2, m, n = num2, num1, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2
    while imin <= imax:
        i = int((imin + imax) / 2)
        j = int(half_len - i)
        # i is too small, must increase it
        if i < m and num2[j - 1] > num1[i]:
            imin = i + 1
        # i is too big, must decrease it
        elif i > 0 and num1[i - 1] > num2[j]:
            imax = i - 1
        # i is perfect
        else:
            if i == 0:
                max_of_left = num2[j - 1]
            elif j == 0:
                max_of_left = num1[i - 1]
            else:
                max_of_left = max(num1[i - 1], num2[j - 1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m:
                min_of_right = num2[j]
            elif j == n:
                min_of_right = num1[i]
            else:
                min_of_right = min(num1[i], num2[j])

            return (max_of_left + min_of_right) / 2.0


# 5
def longestPalindrome(s: str) -> str:
    """
    Given a string s, find the longest palindromic substring in s.
    You may assume that the maximum length of s is 1000.
    """

    def checkCenter(str, l, r):
        """
        Looks for palindromes between l and r
        """
        L, R = l, r
        while L >= 0 and R < len(str) and str[L] == str[R]:
            L -= 1
            R += 1
        return R - L - 1

    if not s or len(s) == 0:
        return ""
    start = end = 0
    for i in range(len(s)):
        len1, len2 = checkCenter(s, i, i), checkCenter(s, i, i + 1)
        maxLen = max(len1, len2)
        if maxLen > end - start:
            start = i - int((maxLen - 1) / 2)
            end = i + int(maxLen / 2)
    return s[start:end + 1]


# 5
def longestPalindromeDP(s: str) -> str:
    dp = [[0] * len(s) for _ in range(len(s))]
    ans = ""
    max_length = 0
    for i in range(len(s) - 1, -1, -1):
        for j in range(i, len(s)):
            if s[i] == s[j] and (j - i < 3 or dp[i + 1][j - 1] == 1):
                dp[i][j] = 1
                if ans == "" or max_length < j - i + 1:
                    ans = s[i:j + 1]
                    max_length = j - i + 1
    return ans


# 6
def zig_zag_conversion(s: str, numRows: int) -> str:
    """
    Runtime: 48 ms, faster than 99.52% of Python3 online submissions for ZigZag Conversion.
    Memory Usage: 13.8 MB, less than 10.00% of Python3 online submissions for ZigZag Conversion.

    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:
    (you may want to display this pattern in a fixed font for better legibility)

    P   A   H   N
    A P L S I I G
    Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"

    Write the code that will take a string and make this conversion given a number of rows:

    :type s: str
    :type numRows: int
    :rtype: str
    """
    if numRows < 2:
        return s
    pos = 1
    step = 1
    lines = {}
    for char in s:
        if pos not in lines:
            lines[pos] = char
        else:
            lines[pos] += char
        pos += step
        if pos == 1 or pos == numRows:
            step *= -1
    sol = ''.join(lines.values())
    return sol


# 7
def reverse_integer(x: int) -> int:
    """
    Given a 32-bit signed integer, reverse digits of an integer.
    Handle 32 bit overflow.

    Runtime: 40 ms, faster than 99.95% of Python3 online submissions for Reverse Integer.
    Memory Usage: 13.2 MB, less than 5.71% of Python3 online submissions for Reverse Integer.

    :param x:
    :return:
    """
    a = int(str(x)[::-1]) if x > 0 else -1 * int(str(x * -1)[::-1])
    return a if -2 ** 31 <= a <= 2 ** 31 - 1 else 0


def myAtoi(str: str) -> int:
    """
    Implement atoi which converts a string to an integer.

    The function first discards as many whitespace characters as necessary until the first non-whitespace character is
    found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical
     digits as possible, and interprets them as a numerical value.The string can contain additional characters after
     those that form the integral number, which are ignored and have no effect on the behavior of this function.
    If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists
    because either str is empty or it contains only whitespace characters, no conversion is performed.
    If no valid conversion could be performed, a zero value is returned.

    Note:
    Only the space character ' ' is considered as whitespace character.
    Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range:
    [−2^31,  2^31 − 1]. If the numerical value is out of the range of representable values, INT_MAX (2^31 − 1) or
    INT_MIN (−2^31) is returned.

    Runtime: 44 ms, faster than 46.26% of Python3 online submissions for String to Integer (atoi).
    Memory Usage: 13.7 MB, less than 5.95% of Python3 online submissions for String to Integer (atoi).
    """
    if not str or len(str) < 1:
        return 0

    INT_MAX = 2 ** 31 - 1
    INT_MIN = -2 ** 31

    i = 0
    while str[i] == ' ' and i < len(str) - 1:
        i += 1
    str = str[i:]
    j = 0
    sign = '+'
    if str[0] == '-':
        sign = '-'
        j += 1
    elif str[0] == '+':
        j += 1

    result = 0
    while len(str) > j and '0' <= str[j] <= '9':
        result *= 10
        result += ord(str[j]) - ord('0')
        j += 1
    if sign == '-':
        result = -result
    if result > INT_MAX:
        return INT_MAX
    elif result < INT_MIN:
        return INT_MIN
    else:
        return result


# 9
def palindrome_number(x: int) -> bool:
    """
    Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.
    Follow up: Could you solve it without converting the integer to a string?

    Runtime: 64 ms, faster than 86.05% of Python3 online submissions for Palindrome Number.
    Memory Usage: 13.8 MB, less than 6.50% of Python3 online submissions for Palindrome Number.
    """
    # If we don't care about the follow up then:
    # return False if x < 0 else x == int(str(x)[::-1])

    # Special cases:
    # As discussed above, when x < 0, x is not a palindrome.
    # Also if the last digit of the number is 0, in order to be a palindrome,
    # the first digit of the number also needs to be 0.
    # Only 0 satisfy this property.
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    res = 0
    while x > res:
        res *= 10
        res += x % 10
        x //= 10
    # When the length is an odd number, we can get rid of the middle digit by revertedNumber/10
    # For example when the input is 12321, at the end of the while loop we get x = 12, revertedNumber = 123,
    # since the middle digit doesn't matter in palindrome(it will always equal to itself), we can simply get rid of it.
    return x == res or x == res // 10


# 10
def regex_match(s: str, p: str) -> bool:
    """
    Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.

    Intuition

    As the problem has an optimal substructure, it is natural to cache intermediate results. We ask the question 
    dp(i, j): does text[i:]text[i:] and pattern[j:]pattern[j:] match? We can
    describe our answer in terms of answers to questions involving smaller strings.

    Algorithm
    We proceed with the same recursion as in Approach 1, except because calls will only ever be made to
    match(text[i:],pattern[j:]), we use dp(i, j) to handle those calls instead, saving us expensive string-building
    operations and allowing us to cache the intermediate results.
    Top-Down
    """

    def dp(i, j, memo):
        if (i, j) not in memo:
            if j == len(p):
                ans = i == len(s)
            else:
                first_match = i < len(s) and p[j] in {s[i], '.'}
                if j + 1 < len(p) and p[j + 1] == '*':
                    ans = dp(i, j + 2, memo) or first_match and dp(i + 1, j, memo)
                else:
                    ans = first_match and dp(i + 1, j + 1, memo)
            memo[i, j] = ans
        return memo[i, j]

    return dp(0, 0, {})


# 10
def regex_match_bottom_up(s: str, p: str) -> bool:
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    dp[-1][-1] = True  # Bottom right
    for i in range(len(s), -1, -1):
        for j in range(len(p) - 1, -1, -1):
            first_match = i < len(s) and p[j] in {s[i], '.'}
            if j + 1 < len(p) and p[j + 1] == '*':
                dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
            else:
                dp[i][j] = first_match and dp[i + 1][j + 1]
    return dp[0][0]


# 11
def container_most_water(height) -> int:
    """
    Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical
    lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together
    with x-axis forms a container, such that the container contains the most water.
    Note: You may not slant the container and n is at least 2.

    Initially we consider the area constituting the exterior most lines. Now, to maximize the area, we need to
    consider the area between the lines of larger lengths. If we try to move the pointer at the longer line inwards,
    we won't gain any increase in area, since it is limited by the shorter line. But moving the shorter line's
    pointer could turn out to be beneficial, as per the same argument, despite the reduction in the width. This is
    done since a relatively longer line obtained by moving the shorter line's pointer might overcome the reduction in
    area caused by the width reduction.
    """
    curMax = 0
    r = len(height) - 1
    l = 0
    while r != l:
        if height[r] > height[l]:
            area = height[l] * (r - l)
            l += 1
        else:
            area = height[r] * (r - l)
            r -= 1
        curMax = max(curMax, area)
    return curMax


# 12
def integer_to_roman(num: int) -> str:
    """
    Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

    Symbol       Value
    I             1
    V             5
    X             10
    L             50
    C             100
    D             500
    M             1000
    For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

    Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

    I can be placed before V (5) and X (10) to make 4 and 9.
    X can be placed before L (50) and C (100) to make 40 and 90.
    C can be placed before D (500) and M (1000) to make 400 and 900.
    Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.


    Runtime: 60 ms, faster than 52.25% of Python3 online submissions for Integer to Roman.
    Memory Usage: 14 MB, less than 6.15% of Python3 online submissions for Integer to Roman.
    """
    if not num:
        return ''
    out = ''
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    Roman = ["M", "CM", "D", "CD", "C", "XC",
             "L", "XL", "X", "IX", "V", "IV", "I"]
    for i in range(len(values)):
        while num >= values[i]:
            num -= values[i]
            out += Roman[i]
    return out


# 13
def roman_to_integer(s: str) -> int:
    """
    Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

    Symbol       Value
    I             1
    V             5
    X             10
    L             50
    C             100
    D             500
    M             1000
    For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

    Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

    I can be placed before V (5) and X (10) to make 4 and 9.
    X can be placed before L (50) and C (100) to make 40 and 90.
    C can be placed before D (500) and M (1000) to make 400 and 900.
    Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.

    Runtime: 52 ms, faster than 80.32% of Python3 online submissions for Roman to Integer.
    Memory Usage: 14 MB, less than 5.38% of Python3 online submissions for Roman to Integer.
    """
    roman = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    res = 0
    for i in range(0, len(s) - 1):
        res += -roman[s[i]] if roman[s[i]] < roman[s[i + 1]] else roman[s[i]]
    res += roman[s[-1]]
    return res


# 14
def longest_common_prefix(strs) -> str:
    """
    Write a function to find the longest common prefix string amongst an array of strings.
    If there is no common prefix, return an empty string "".

    Runtime: 32 ms, faster than 98.37% of Python3 online submissions for Longest Common Prefix.
    Memory Usage: 13.8 MB, less than 6.67% of Python3 online submissions for Longest Common Prefix.
    """
    if not strs:
        return ""
    shortest = min(strs, key=len)
    strs.remove(shortest)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch:
                return shortest[:i]
    return shortest


# 15
def threeSum(nums: List[int]) -> List[List[int]]:
    """
    Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique
    triplets in the array which gives the sum of zero.
    Note: The solution set must not contain duplicate triplets.

    Runtime: 636 ms, faster than 95.48% of Python3 online submissions for 3Sum.
    Memory Usage: 18.6 MB, less than 5.71% of Python3 online submissions for 3Sum.
    """
    if len(nums) < 3:
        return []
    nums.sort()
    triplets = set()
    for idx, pivot in enumerate(nums[:-2]):
        if pivot <= 0 and (idx == 0 or pivot > nums[idx - 1]):
            d = {}
            for number in nums[idx + 1:]:
                remaining = - pivot - number
                if number not in d:
                    d[remaining] = 1
                else:
                    triplets.add((pivot, number, remaining))
    return [*map(list, triplets)]


# 19
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    """
    Given a linked list, remove the n-th node from the end of list and return its head.
    Example: Given linked list: 1->2->3->4->5, and n = 2.
    After removing the second node from the end, the linked list becomes 1->2->3->5.
    Note: Given n will always be valid.
    Follow up: Could you do this in one pass?
    """
    temp = ListNode(None)  # Create and prepend a new list node
    temp.next = head
    ptr1 = ptr2 = temp  # ptr1 is never ahead of ptr2

    for _ in range(n):
        ptr2 = ptr2.next

    while ptr2.next:
        ptr1 = ptr1.next
        ptr2 = ptr2.next

    ptr1.next = ptr1.next.next
    return temp.next


# 20
def validParenthesis(s: str) -> bool:
    """
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
    determine if the input string is valid.
    An input string is valid if:
    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    Note that an empty string is also considered valid.

    Runtime: 32 ms, faster than 94.70% of Python3 online submissions for Valid Parentheses.
    Memory Usage: 13.8 MB, less than 5.22% of Python3 online submissions for Valid Parentheses.
    """
    syms = {'(': ')', '{': '}', '[': ']'}
    stack = []
    for c in s:
        if c in syms.keys():
            stack.append(c)
        elif c in syms.values() and stack:
            p = stack.pop()
            if syms[p] != c:
                return False
        else:
            return False
    return not stack


# 21
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
    Example:
    Input: 1->2->4, 1->3->4
    Output: 1->1->2->3->4->4
    """
    head = current = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 if l1 else l2
    return head.next


# 22. Generate Parentheses

def generateParenthesis(n: int) -> List[str]:
    """"
    Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
    For example, given n = 3, a solution set is:
    [
      "((()))",
      "(()())",
      "(())()",
      "()(())",
      "()()()"
    ]

    Runtime: 24 ms, faster than 99.25% of Python3 online submissions for Generate Parentheses.
    Memory Usage: 12.9 MB, less than 100.00% of Python3 online submissions for Generate Parentheses.
    """

    def backtrack(ans, seq='', l=0, r=0):
        # Check for validation only at 2*n elements
        if len(seq) == 2 * n:
            ans.append(seq)
            return
        # Try to add more of the missing parenthesis
        if l < n:
            backtrack(ans, seq + '(', l + 1, r)
        if r < l:
            backtrack(ans, seq + ')', l, r + 1)

    ans = []
    backtrack(ans)
    return ans


def mergeKLists(lists: List[ListNode]) -> ListNode:
    """
    Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

    This approach walks alongside the one above but is improved a lot. We don't need to traverse most nodes many times repeatedly
    Pair up k lists and merge each pair.
    After the first pairing, k lists are merged into k/2 lists with average 2N/k length, then k/4, k/8 and so on.
    Repeat this procedure until we get the final sorted linked list.
    Thus, we'll traverse almost N nodes per pairing and merging, and repeat this procedure about log_{2}{k} times.

    Example:
    Input:
    [
      1->4->5,
      1->3->4,
      2->6
    ]
    Output: 1->1->2->3->4->4->5->6

    Runtime: 160 ms, faster than 29.31% of Python3 online submissions for Merge k Sorted Lists.
    Memory Usage: 16 MB, less than 100.00% of Python3 online submissions for Merge k Sorted Lists.
    """

    def merge(l, r):
        head = cur = ListNode(0)
        while l and r:
            if l.val < r.val:
                cur.next = l
                l = l.next
            else:
                cur.next = r
                r = r.next
            cur = cur.next
        cur.next = l or r
        return head.next

    # Edge cases
    if not lists:
        return None  # Needs to be None
    elif len(lists) == 1:
        return lists[0]

    # Divide the list of lists into 2 equal sub-lists and merge each sublist recursively
    mid = len(lists) // 2
    l = mergeKLists(lists[:mid])
    r = mergeKLists(lists[mid:])
    return merge(l, r)


# 26. Remove Duplicates from Sorted Array
def removeDuplicates(nums: List[int]) -> int:
    """
    Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
    Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
    Example 1:
    Given nums = [1,1,2],
    Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.
    It doesn't matter what you leave beyond the returned length.

    Example 2:
    Given nums = [0,0,1,1,1,2,2,3,3,4],
    Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, and 4 respectively.
    It doesn't matter what values are set beyond the returned length.
    Clarification:
    Confused why the returned value is an integer but your answer is an array?
    Note that the input array is passed in by reference, which means modification to the input array will be known to the caller as well.

    Internally you can think of this:
    // nums is passed in by reference. (i.e., without making a copy)
    int len = removeDuplicates(nums);
    // any modification to nums in your function would be known by the caller.
    // using the length returned by your function, it prints the first len elements.
    for (int i = 0; i < len; i++) {
        print(nums[i]);
    }

    Runtime: 76 ms, faster than 99.18% of Python3 online submissions for Remove Duplicates from Sorted Array.
    Memory Usage: 14.5 MB, less than 98.36% of Python3 online submissions for Remove Duplicates from Sorted Array.
    """
    if len(nums) < 1:
        return 0
    unq_ptr, unq_val = 1, nums[0]
    for cur_idx, cur_val in enumerate(nums):
        if cur_val != unq_val:
            unq_val = nums[unq_ptr] = cur_val
            unq_ptr += 1
    return unq_ptr


# 28. Implement strStr()
def strStr(haystack: str, needle: str) -> int:
    """
    Implement strStr().
    Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

    Example 1:
    Input: haystack = "hello", needle = "ll"
    Output: 2

    Example 2:
    Input: haystack = "aaaaa", needle = "bba"
    Output: -1
    Clarification:

    What should we return when needle is an empty string? This is a great question to ask during an interview.
    For the purpose of this problem, we will return 0 when needle is an empty string.
    This is consistent to C's strstr() and Java's indexOf().

    Runtime: 24 ms, faster than 97.35% of Python3 online submissions for Implement strStr().
    Memory Usage: 12.8 MB, less than 100.00% of Python3 online submissions for Implement strStr().
    """
    if not needle:
        return 0
    len_n, len_h = len(needle), len(haystack)
    if len_n > len_h:
        return -1
    for idx_haystack in range(len_h - len_n + 1):
        for idx_needle in range(len_n):
            if haystack[idx_haystack + idx_needle] != needle[idx_needle]:
                break
        else:
            return idx_haystack
    return -1


# 29. Divide Two Integers
def divide(dividend: int, divisor: int) -> int:
    """
    Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.
    Return the quotient after dividing dividend by divisor.

    The integer division should truncate toward zero.

    Example 1:
    Input: dividend = 10, divisor = 3
    Output: 3

    Example 2:
    Input: dividend = 7, divisor = -3
    Output: -2
    Note:

    Both dividend and divisor will be 32-bit signed integers.
    The divisor will never be 0.
    Assume we are dealing with an environment which could only store integers within the 32-bit signed integer
    range: [−2^31,  2^31 − 1]. For the purpose of this problem, assume that your function returns 2^31 − 1 when
    the division result overflows.

    Runtime: 28 ms, faster than 94.12% of Python3 online submissions for Divide Two Integers.
    Memory Usage: 12.8 MB, less than 100.00% of Python3 online submissions for Divide Two Integers.

    Runtime: 24 ms, faster than 98.34% of Python3 online submissions for Divide Two Integers.
    Memory Usage: 12.7 MB, less than 100.00% of Python3 online submissions for Divide Two Integers.
    """
    abs_dividend, abs_divisor, res, sign = abs(dividend), abs(divisor), 0, (dividend > 0) == (divisor > 0)
    while abs_dividend >= abs_divisor:
        x = 0
        while abs_dividend >= abs_divisor << (x + 1):
            x += 1
        res += 1 << x
        abs_dividend -= abs_divisor << x
    return min(res if sign else -res, pow(2, 31) - 1)


# 33. Search in Rotated Sorted Array
def search(nums: List[int], target: int) -> int:
    """
    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
    (i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

    You are given a target value to search. If found in the array return its index, otherwise return -1.
    You may assume no duplicate exists in the array.
    Your algorithm's runtime complexity must be in the order of O(log n).

    Example 1:
    Input: nums = [4,5,6,7,0,1,2], target = 0
    Output: 4

    Example 2:
    Input: nums = [4,5,6,7,0,1,2], target = 3
    Output: -1

    Runtime: 40 ms, faster than 88.18% of Python3 online submissions for Search in Rotated Sorted Array.
    Memory Usage: 12.9 MB, less than 100.00% of Python3 online submissions for Search in Rotated Sorted Array.
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= target < nums[mid] or target < nums[mid] < nums[lo] or nums[mid] < nums[lo] <= target:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


# 42. Trapping Rain Water
def trap_brute_force(height: List[int]) -> int:
    """
    Given n non-negative integers representing an elevation map where the width of each bar is 1,
    compute how much water it is able to trap after raining.
    Example:
        Input: [0,1,0,2,1,0,1,3,2,1,2,1]
        Output: 6
    """
    if not height or len(height) < 3:
        return 0
    ans = 0
    for pos, h in enumerate(height):
        max_left = max(height[:pos], default=0)
        max_right = max(height[pos:], default=0)
        ans += max(0, min(max_left, max_right) - h)
    return ans


# 42
def trap_2P(height):
    """
    Runtime: 56 ms, faster than 44.10% of Python3 online submissions for Trapping Rain Water.
    Memory Usage: 13.5 MB, less than 65.12% of Python3 online submissions for Trapping Rain Water.
    """
    if not height or len(height) < 3:
        return 0
    l, r = 0, len(height) - 1
    l_max, r_max = height[l], height[r]
    ans = 0
    while l < r:
        l_max = max(l_max, height[l])
        r_max = max(r_max, height[r])
        if l_max <= r_max:
            ans += (l_max - height[l])
            l += 1
        else:
            ans += (r_max - height[r])
            r -= 1
    return ans


# 42
def trap_DP(height: List[int]) -> int:
    if not height or len(height) < 3:
        return 0

    n = len(height)

    left_dp = [0 for _ in range(n)]
    right_dp = [0 for _ in range(n)]
    left_dp[0] = height[0]
    right_dp[-1] = height[-1]

    # dp array from left to right record max height so far
    for i in range(1, n):
        left_dp[i] = max(left_dp[i - 1], height[i])

    # dp array from right to left record max height so far
    for i in range(n - 2, -1, -1):
        right_dp[i] = max(right_dp[i + 1], height[i])

    res = 0
    for pos, h in enumerate(height):
        res += min(left_dp[pos], right_dp[pos]) - h
    return res


# 42
def trap_Stack(height: List[int]) -> int:
    decreasingHeightStack, totalWaterTrapped = [], 0
    for i, v in enumerate(height):
        while decreasingHeightStack and height[decreasingHeightStack[-1]] < v:
            bottomHeight = height[decreasingHeightStack.pop()]
            if not decreasingHeightStack:
                break
            leftUpperIndex = decreasingHeightStack[-1]
            heightDiff = min(height[leftUpperIndex], v) - bottomHeight
            width = i - leftUpperIndex - 1
            totalWaterTrapped += heightDiff * width
        decreasingHeightStack.append(i)
    return totalWaterTrapped


# 44. Wildcard Matching
def wildcard_match(s: str, p: str) -> bool:
    """
    Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.
    '?' Matches any single character.
    '*' Matches any sequence of characters (including the empty sequence).
    The matching should cover the entire input string (not partial).
    Note:
    s could be empty and contains only lowercase letters a-z.
    p could be empty and contains only lowercase letters a-z, and characters like ? or *.
    """
    s_itr = p_itr = 0
    ls = len(s)
    lp = len(p)
    star = s_idx = -1
    while s_itr < ls:
        if p_itr < lp and (s[s_itr] == p[p_itr] or p[p_itr] == "?"):
            p_itr += 1
            s_itr += 1
        elif p_itr < lp and p[p_itr] == "*":
            star = p_itr
            s_idx = s_itr
            p_itr += 1  # I just take the "*" zero times
        elif star != -1:
            p_itr = star + 1  # star at star eats all the char before and at s_idx
            s_idx += 1  # i eat the char at s_idx
            s_itr = s_idx
        else:
            return False

    while p_itr < lp and p[p_itr] == "*":
        p_itr += 1

    return p_itr == lp


# 44
def wildcard_match_DP(s: str, p: str) -> bool:
    m = len(s)
    n = len(p)

    if n == 0 and m == 0:
        return True
    if n == 0 and m != 0:
        return False
    if p.count('*') == len(p):
        return True
    if len(p) - p.count('*') > m:
        return False

    dp = [True] + [False] * m
    for cp in p:
        if cp != '*':
            for i in range(m, 0, -1):  # m, m-1, .. 1
                dp[i] = dp[i - 1] and (cp == '?' or s[i - 1] == cp)
        else:
            for i in range(1, m + 1):
                dp[i] = dp[i] or dp[i - 1]
        dp[0] = dp[0] and cp == '*'

    return dp[m]


# 46. Permutations
def permute_dfs(nums: List[int]) -> List[List[int]]:
    """
    Given a collection of distinct integers, return all possible permutations.

    Example:

    Input: [1,2,3]
    Output:
    [
      [1,2,3],
      [1,3,2],
      [2,1,3],
      [2,3,1],
      [3,1,2],
      [3,2,1]
    ]
    """

    def dfs(nums, path, res):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            dfs(nums[:i] + nums[i + 1:], path + [nums[i]], res)

    res = []
    dfs(nums, [], res)
    return res


# 46
def permute_reduce(nums: List[int]) -> List[List[int]]:
    return reduce(lambda P, n: [p[:i] + [n] + p[i:] for p in P for i in range(len(p) + 1)], nums, [[]])


# 48
def rotate(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    You are given an n x n 2D matrix representing an image.

    Rotate the image by 90 degrees (clockwise).

    Note:

    You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
    DO NOT allocate another 2D matrix and do the rotation.

    Runtime: 36 ms, faster than 39.35% of Python3 online submissions for Rotate Image.
    Memory Usage: 13 MB, less than 95.83% of Python3 online submissions for Rotate Image.
    """
    n = len(matrix)
    for i in range(floor(n / 2)):
        for j in range(n - floor(n / 2)):
            matrix[i][j], matrix[~j][i], matrix[~i][~j], matrix[j][~i] = \
                matrix[~j][i], matrix[~i][~j], matrix[j][~i], matrix[i][j]


# 48
def rotate_B(matrix: List[List[int]]) -> None:
    n = len(matrix)
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        for j in range(int(n / 2)):
            row[j], row[~j] = row[~j], row[j]


# 49. Group Anagrams
def groupAnagrams(strs: List[str]) -> List[List[str]]:
    """
    Given an array of strings, group anagrams together.

    Example:

    Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
    Output:
    [
      ["ate","eat","tea"],
      ["nat","tan"],
      ["bat"]
    ]
    Note:

    All inputs will be in lowercase.
    The order of your output does not matter.

    Runtime: 100 ms, faster than 72.53% of Python3 online submissions for Group Anagrams.
    Memory Usage: 16.5 MB, less than 45.28% of Python3 online submissions for Group Anagrams.
    """
    d = {}
    for w in strs:
        key = tuple(sorted(w))
        d[key] = d.get(key, []) + [w]
    return list(d.values())


# 50. Pow(x, n)
def myPow(x, n):
    """
    Implement pow(x, n), which calculates x raised to the power n (xn).

    Example 1:

    Input: 2.00000, 10
    Output: 1024.00000
    Example 2:

    Input: 2.10000, 3
    Output: 9.26100
    Example 3:

    Input: 2.00000, -2
    Output: 0.25000
    Explanation: 2-2 = 1/22 = 1/4 = 0.25
    Note:

    -100.0 < x < 100.0
    n is a 32-bit signed integer, within the range [−231, 231 − 1]
    """
    if not n:
        return 1
    if n < 0:
        return 1 / myPow(x, -n)
    if n % 2:
        return x * myPow(x, n - 1)
    return myPow(x * x, n / 2)


# 53. Maximum Subarray
def maxSubArray(nums: List[int]) -> int:
    """
    Given an integer array nums, find the contiguous subarray
    (containing at least one number) which has the largest sum and
     return its sum.

    Example:
    Input: [-2,1,-3,4,-1,2,1,-5,4],
    Output: 6
    Explanation: [4,-1,2,1] has the largest sum = 6.
    Follow up:

    If you have figured out the O(n) solution, try coding another solution using the divide and conquer
    approach, which is more subtle.
    """
    for i in range(1, len(nums)):
        nums[i] = max(nums[i], nums[i - 1] + nums[i])
    return max(nums)


def letter_combinations(digits: str) -> List[str]:
    if not digits:
        return []

    buttons = {'2': ['a', 'b', 'c'],
               '3': ['d', 'e', 'f'],
               '4': ['g', 'h', 'i'],
               '5': ['j', 'k', 'l'],
               '6': ['m', 'n', 'o'],
               '7': ['p', 'q', 'r', 's'],
               '8': ['t', 'u', 'v'],
               '9': ['w', 'x', 'y', 'z']}

    def backtrack(combination, next_digits, output):
        # if there is no more digits to check
        if len(next_digits) == 0:
            # the combination is done
            output.append(combination)
        # if there are still digits to check
        else:
            # iterate over all letters which map the next available digit
            for letter in buttons[next_digits[0]]:
                # append the current letter to the combination and proceed to the next digits
                backtrack(combination + letter, next_digits[1:], output)

    output = []
    backtrack("", digits, output)
    return output


def letter2(digits: str) -> List[str]:
    s = "abcdefghijklmnopqrstuvwxyz"
    dtos = {}
    if digits == "":
        return []
    for i in range(5):
        dtos[i + 2] = s[i * 3:i * 3 + 3]
    dtos[7] = 'pqrs'
    dtos[8] = 'tuv'
    dtos[9] = 'wxyz'
    res = []
    for d in digits:
        res.append(dtos[int(d)])

    def dp(i):  # res[i:]的组合
        result = []
        if i == len(res):
            return ['']
        chars = res[i]
        for c in chars:
            chars_next = dp(i + 1)
            for cn in chars_next:
                result.append(c + cn)
        return result

    return dp(0)


# 739
def daily_temperature(T: List[int]) -> List[int]:
    """
    Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days
    you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.
    For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].
    Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].

    Runtime: 512 ms, faster than 98.30% of Python3 online submissions for Daily Temperatures.
    Memory Usage: 17.6 MB, less than 7.89% of Python3 online submissions for Daily Temperatures.
    """
    ans = [0] * len(T)
    stack = []
    for i in range(len(T)):
        while stack and T[stack[-1]] < T[i]:
            prev_i = stack.pop()
            ans[prev_i] = i - prev_i
        stack.append(i)
    return ans
