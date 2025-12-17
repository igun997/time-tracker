from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TimeEntry:
    """Represents a time entry for reporting"""
    employee_id: int
    employee_name: str
    date: date
    total_seconds: int
    source_name: Optional[str] = None

    @property
    def hours(self) -> int:
        return self.total_seconds // 3600

    @property
    def minutes(self) -> int:
        return (self.total_seconds % 3600) // 60

    @property
    def seconds(self) -> int:
        return self.total_seconds % 60

    @property
    def formatted_duration(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}"

    @property
    def decimal_hours(self) -> float:
        return round(self.total_seconds / 3600, 2)

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "date": self.date.isoformat(),
            "total_seconds": self.total_seconds,
            "formatted_duration": self.formatted_duration,
            "decimal_hours": self.decimal_hours,
            "source_name": self.source_name
        }


@dataclass
class DailyReport:
    """Daily time tracking report"""
    report_date: date
    entries: List[TimeEntry]

    @property
    def total_seconds(self) -> int:
        return sum(e.total_seconds for e in self.entries)

    @property
    def total_formatted(self) -> str:
        total = self.total_seconds
        hours = total // 3600
        minutes = (total % 3600) // 60
        return f"{hours:02d}:{minutes:02d}"

    @property
    def employee_count(self) -> int:
        return len(set(e.employee_id for e in self.entries))

    def to_dict(self) -> dict:
        return {
            "date": self.report_date.isoformat(),
            "total_seconds": self.total_seconds,
            "total_formatted": self.total_formatted,
            "employee_count": self.employee_count,
            "entries": [e.to_dict() for e in self.entries]
        }


@dataclass
class EmployeeReport:
    """Time tracking report for a single employee"""
    employee_id: int
    employee_name: str
    start_date: date
    end_date: date
    daily_entries: List[TimeEntry]

    @property
    def total_seconds(self) -> int:
        return sum(e.total_seconds for e in self.daily_entries)

    @property
    def total_formatted(self) -> str:
        total = self.total_seconds
        hours = total // 3600
        minutes = (total % 3600) // 60
        return f"{hours:02d}:{minutes:02d}"

    @property
    def average_daily_seconds(self) -> int:
        if not self.daily_entries:
            return 0
        return self.total_seconds // len(self.daily_entries)

    @property
    def average_daily_formatted(self) -> str:
        avg = self.average_daily_seconds
        hours = avg // 3600
        minutes = (avg % 3600) // 60
        return f"{hours:02d}:{minutes:02d}"

    @property
    def days_present(self) -> int:
        return len(set(e.date for e in self.daily_entries))

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "employee_name": self.employee_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_seconds": self.total_seconds,
            "total_formatted": self.total_formatted,
            "average_daily_formatted": self.average_daily_formatted,
            "days_present": self.days_present,
            "daily_entries": [e.to_dict() for e in self.daily_entries]
        }


class TimeCalculator:
    """Utility class for time calculations and formatting"""

    @staticmethod
    def seconds_to_hms(seconds: int) -> tuple:
        """Convert seconds to (hours, minutes, seconds)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return (hours, minutes, secs)

    @staticmethod
    def format_duration(seconds: int, include_seconds: bool = True) -> str:
        """Format seconds as HH:MM or HH:MM:SS"""
        hours, minutes, secs = TimeCalculator.seconds_to_hms(seconds)
        if include_seconds:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{hours:02d}:{minutes:02d}"

    @staticmethod
    def format_duration_human(seconds: int) -> str:
        """Format duration in human-readable format"""
        hours, minutes, secs = TimeCalculator.seconds_to_hms(seconds)
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)

    @staticmethod
    def to_decimal_hours(seconds: int) -> float:
        """Convert seconds to decimal hours"""
        return round(seconds / 3600, 2)

    @staticmethod
    def aggregate_by_employee(entries: List[TimeEntry]) -> Dict[int, TimeEntry]:
        """Aggregate time entries by employee"""
        aggregated: Dict[int, TimeEntry] = {}

        for entry in entries:
            if entry.employee_id in aggregated:
                aggregated[entry.employee_id].total_seconds += entry.total_seconds
            else:
                aggregated[entry.employee_id] = TimeEntry(
                    employee_id=entry.employee_id,
                    employee_name=entry.employee_name,
                    date=entry.date,
                    total_seconds=entry.total_seconds
                )

        return aggregated

    @staticmethod
    def aggregate_by_date(entries: List[TimeEntry]) -> Dict[date, List[TimeEntry]]:
        """Group time entries by date"""
        by_date: Dict[date, List[TimeEntry]] = {}

        for entry in entries:
            if entry.date not in by_date:
                by_date[entry.date] = []
            by_date[entry.date].append(entry)

        return by_date

    @staticmethod
    def get_date_range(start_date: date, end_date: date) -> List[date]:
        """Generate list of dates in range"""
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    @staticmethod
    def calculate_weekly_summary(entries: List[TimeEntry]) -> dict:
        """Calculate weekly summary statistics"""
        if not entries:
            return {
                "total_seconds": 0,
                "total_formatted": "00:00",
                "average_daily": "00:00",
                "days_worked": 0
            }

        total = sum(e.total_seconds for e in entries)
        unique_dates = len(set(e.date for e in entries))

        return {
            "total_seconds": total,
            "total_formatted": TimeCalculator.format_duration(total, False),
            "average_daily": TimeCalculator.format_duration(
                total // unique_dates if unique_dates else 0, False
            ),
            "days_worked": unique_dates
        }
