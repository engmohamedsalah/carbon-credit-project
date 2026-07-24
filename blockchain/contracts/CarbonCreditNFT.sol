// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract CarbonCreditNFT is ERC721, ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter;
    
    struct CarbonCredit {
        uint256 projectId;
        uint256 carbonAmount; // in tCO2e
        string projectName;
        string location;
        uint256 verificationDate;
        string verificationHash;
        bool isRetired;
    }
    
    mapping(uint256 => CarbonCredit) public carbonCredits;
    mapping(uint256 => bool) public projectExists;
    
    event CarbonCreditMinted(
        uint256 indexed tokenId,
        uint256 indexed projectId,
        uint256 carbonAmount,
        address recipient
    );
    
    event CarbonCreditRetired(uint256 indexed tokenId);
    
    constructor() ERC721("Carbon Credit Certificate", "CARBON") {}
    
    function mintCarbonCredit(
        address to,
        uint256 projectId,
        uint256 carbonAmount,
        string memory projectName,
        string memory location,
        string memory verificationHash,
        string memory tokenURI
    ) public onlyOwner returns (uint256) {
        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();
        
        carbonCredits[tokenId] = CarbonCredit({
            projectId: projectId,
            carbonAmount: carbonAmount,
            projectName: projectName,
            location: location,
            verificationDate: block.timestamp,
            verificationHash: verificationHash,
            isRetired: false
        });
        
        projectExists[projectId] = true;
        
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);
        
        emit CarbonCreditMinted(tokenId, projectId, carbonAmount, to);
        
        return tokenId;
    }
    
    function retireCarbonCredit(uint256 tokenId) public {
        require(_exists(tokenId), "Token does not exist");
        require(ownerOf(tokenId) == msg.sender || owner() == msg.sender, "Not authorized");
        require(!carbonCredits[tokenId].isRetired, "Already retired");
        
        carbonCredits[tokenId].isRetired = true;
        emit CarbonCreditRetired(tokenId);
    }
    
    function getCarbonCredit(uint256 tokenId) public view returns (CarbonCredit memory) {
        require(_exists(tokenId), "Token does not exist");
        return carbonCredits[tokenId];
    }
    
    function totalSupply() public view returns (uint256) {
        return _tokenIdCounter.current();
    }
    
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}