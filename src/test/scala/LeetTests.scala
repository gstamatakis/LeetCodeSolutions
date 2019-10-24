import LeetCode._
import org.scalatest.FunSuite

class LeetTests extends FunSuite {
	//1
	test("TwoSum") {
		assert(twoSum(Array(), 999) === Array())
		assert(twoSum(Array(1, 2), 3) === Array(0, 1))
		assert(twoSum(Array(2, 7, 11, 15), 9) === Array(0, 1))
	}

	//No 5
	test("longestPalindrome") {
		assert(longestPalindrome("babad") === "bab")
		assert(longestPalindrome("cbbd") === "bb")
	}

	//118
	test("Pascal's Triangle") {
		assert(pascalTriangle(5) ===
			List(List(1), List(1, 1), List(1, 2, 1), List(1, 3, 3, 1), List(1, 4, 6, 4, 1)))
		assert(pascalTriangle(1) === List(List(1)))
		assert(pascalTriangle(500).size == 500)
	}

}