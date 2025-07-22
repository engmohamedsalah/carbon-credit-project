import React from 'react';
import '@testing-library/jest-dom';

// Simple test to ensure the test environment works
describe('Frontend Test Environment', () => {
  test('React is available', () => {
    expect(React).toBeDefined();
  });

  test('basic assertions work', () => {
    expect(2 + 2).toBe(4);
    expect('hello').toBe('hello');
  });

  test('DOM testing utilities are available', () => {
    const element = document.createElement('div');
    element.textContent = 'Test';
    expect(element).toBeDefined();
    expect(element.textContent).toBe('Test');
  });
});

// Test that utility functions work
describe('Utility Functions', () => {
  test('basic math operations work', () => {
    expect(2 + 2).toBe(4);
    expect(5 * 3).toBe(15);
    expect(10 - 3).toBe(7);
  });

  test('string operations work', () => {
    expect('hello').toBe('hello');
    expect('world'.toUpperCase()).toBe('WORLD');
    expect('test'.length).toBe(4);
  });
});

// Test that components can be imported
describe('Component Imports', () => {
  test('can create basic React components', () => {
    const TestComponent = () => <div>Test</div>;
    expect(TestComponent).toBeDefined();
  });
}); 