import scala.annotation.tailrec
import scala.collection.mutable

object LeetCode extends App {
	/**
	 * No 1
	 *
	 * Given an array of integers, return indices of the two numbers such that they add up to a specific target.
	 * You may assume that each input would have exactly one solution, and you may not use the same element twice.
	 *
	 * Runtime: 264 ms, faster than 99.88% of Scala online submissions for Two Sum.
	 * Memory Usage: 48.5 MB, less than 100.00% of Scala online submissions for Two Sum.
	 *
	 * @param nums   Input array of nums.
	 * @param target The target sum.
	 * @return An array of INDICES of the selected array numbers.
	 */
	def twoSum(nums: Array[Int], target: Int): Array[Int] = {
		if (nums.length == 0 || nums.length == 1) return Array()
		if (nums.length == 2) return Array(0, 1)
		val lookup = mutable.Map[Int, Int]()
		for (i <- nums.indices) {
			val el = nums(i)
			val res = target - el
			if (lookup.contains(el)) {
				return Array(lookup(el), i)
			} else {
				lookup.put(res, i)
			}
		}
		Array()
	}

	/**
	 * No 5
	 *
	 * Given a string s, find the longest palindromic substring in s.
	 * You may assume that the maximum length of s is 1000.
	 *
	 * @param s Input string.
	 * @return A string with the longest palindrome.
	 */
	def longestPalindrome(s: String): String = {
		@tailrec
		def maxAt(startIndex: Int, endIndex: Int): String = {
			if (startIndex < 0 || endIndex > s.length - 1 || s.charAt(startIndex) != s.charAt(endIndex))
				s.substring(startIndex + 1, endIndex)
			else
				maxAt(startIndex - 1, endIndex + 1)
		}

		@tailrec
		def _lp(st: Int, largestYet: String): String = {
			if (st == s.length) largestYet
			else {
				val largestOdd = maxAt(st - 1, st + 1)
				val largestEven = maxAt(st, st + 1)

				if (largestOdd.length > largestEven.length && largestOdd.length > largestYet.length)
					_lp(st + 1, largestOdd)
				else if (largestEven.length > largestOdd.length && largestEven.length > largestYet.length)
					_lp(st + 1, largestEven)
				else
					_lp(st + 1, largestYet)
			}
		}

		_lp(0, "")
	}


	/**
	 * No 118
	 *
	 * Creates a Pascal's Triangle up to the given row number.
	 *
	 * Runtime: 204 ms, faster than 96.59% of Scala online submissions for Pascal's Triangle.
	 * Memory Usage: 42.6 MB, less than 100.00% of Scala online submissions for Pascal's Triangle.
	 *
	 * @param numRows The number of rows
	 * @return A List of Lists with each row representing the Triangle's row at the given depth.
	 **/
	def pascalTriangle(numRows: Int): List[List[Int]] = {
		def constructFrom(prevRow: List[Int]): List[Int] = prevRow match {
			case _ :: Nil => List(1)
			case x1 :: x2 :: xs => (x1 + x2) :: constructFrom(x2 :: xs)
		}

		def genFromPrev(n: Int): List[List[Int]] =
			if (n == 1) {
				List(List(1))
			} else {
				val prev = genFromPrev(n - 1)
				(1 :: constructFrom(prev.head)) :: prev
			}

		if (numRows == 0) Nil else genFromPrev(numRows).reverse
	}

}