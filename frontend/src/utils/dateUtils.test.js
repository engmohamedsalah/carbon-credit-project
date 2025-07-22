import { formatDate, formatDateForInput, isValidDate, getRelativeTime } from './dateUtils';

describe('Date Utils', () => {
  test('formatDate formats date correctly', () => {
    const testDate = '2023-12-25T10:30:00';
    const formatted = formatDate(testDate);
    expect(typeof formatted).toBe('string');
    expect(formatted).toContain('Dec');
  });

  test('formatDateForInput formats date for input fields', () => {
    const testDate = '2023-12-25T10:30:00';
    const formatted = formatDateForInput(testDate);
    expect(typeof formatted).toBe('string');
    expect(formatted).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  test('isValidDate validates dates correctly', () => {
    expect(isValidDate('2023-12-25')).toBe(true);
    expect(isValidDate('invalid-date')).toBe(false);
    expect(isValidDate(null)).toBe(false);
  });

  test('getRelativeTime returns relative time string', () => {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const relative = getRelativeTime(yesterday.toISOString());
    expect(typeof relative).toBe('string');
    expect(relative).toBe('Yesterday');
  });

  test('formatDate handles null input gracefully', () => {
    const formatted = formatDate(null);
    expect(formatted).toBe('Not set');
  });

  test('formatDate handles invalid date gracefully', () => {
    const formatted = formatDate('invalid-date');
    expect(formatted).toBe('Invalid date');
  });
}); 