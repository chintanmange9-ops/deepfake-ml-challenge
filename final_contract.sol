// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title DeepfakeLedgerV4
 * @notice Store deepfake predictions with confidence and full history per image hash.
 *         Uses string imageHash so you can send the hex SHA256 (e.g. "a3f4...") directly.
 */
contract DeepfakeLedgerV4 {
    struct Record {
        string prediction;
        uint256 confidence;
        uint256 timestamp;
        address submittedBy;
    }

    mapping(string => Record[]) private records;

    event RecordStored(
        string indexed imageHash,
        string prediction,
        uint256 confidence,
        uint256 timestamp,
        address indexed submittedBy
    );

    function storeRecord(
        string memory imageHash,
        string memory prediction,
        uint256 confidence
    ) public {
        records[imageHash].push(Record({
            prediction: prediction,
            confidence: confidence,
            timestamp: block.timestamp,
            submittedBy: msg.sender
        }));
        emit RecordStored(imageHash, prediction, confidence, block.timestamp, msg.sender);
    }

    function getRecordCount(string memory imageHash) public view returns (uint256) {
        return records[imageHash].length;
    }

    function getRecord(string memory imageHash, uint256 index)
        public
        view
        returns (string memory, uint256, uint256, address)
    {
        require(index < records[imageHash].length, "Invalid index");
        Record storage r = records[imageHash][index];
        return (r.prediction, r.confidence, r.timestamp, r.submittedBy);
    }

    function getLatestRecord(string memory imageHash)
        public
        view
        returns (string memory, uint256, uint256, address)
    {
        uint256 cnt = records[imageHash].length;
        require(cnt > 0, "No records");
        Record storage r = records[imageHash][cnt - 1];
        return (r.prediction, r.confidence, r.timestamp, r.submittedBy);
    }

}
