//SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData
{
    // Structure
    struct Student
    {
        int ID;
        string fName;
        string lName;
        int[2] marks;     // array of 2 marks
    }

    address owner;
    int public stdCount = 0;

    mapping(int => Student) public stdRecords;

    // Fallback function (required)
    fallback() external payable {
        // contract can receive ETH
    }

    // Receive function (optional but good)
    receive() external payable {}

    modifier onlyOwner {
        require(owner == msg.sender, "Only owner can add");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // Add new student record
    function addNewRecords(
        int _ID,
        string memory _fName,
        string memory _lName,
        int[2] memory _marks
    ) 
        public 
        onlyOwner 
    {
        stdCount = stdCount + 1;
        stdRecords[stdCount] = Student(_ID, _fName, _lName, _marks);
    }
}
