// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Bank {

    // Mapping to store balance of each customer
    mapping(address => uint) public balances;

    // Deposit money into your account
    function deposit() public payable {
        require(msg.value > 0, "Amount must be greater than 0");
        balances[msg.sender] += msg.value;
    }

    // Withdraw money from your account
    function withdraw(uint _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient funds");

        balances[msg.sender] -= _amount;

        // Transfer ETH back to user
        payable(msg.sender).transfer(_amount);
    }

    // Show your balance
    function getBalance() public view returns(uint) {
        return balances[msg.sender];
    }

    // Fallback & receive functions to accept ETH
    receive() external payable {
        balances[msg.sender] += msg.value;
    }

    fallback() external payable {
        balances[msg.sender] += msg.value;
    }
}
