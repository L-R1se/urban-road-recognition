"""
Servo PD Controller for Steering.

Implements a Proportional-Derivative controller that converts lane offset error
into servo angle commands for smooth steering.
"""


class Servo_PD:
    """
    PD controller for servo-based steering.

    output = Kp * error + Kd * (error - last_error)

    Output is clamped to [-output_limit, +output_limit] to prevent
    over-steering on sharp turns.
    """

    def __init__(self, servo_p, servo_d):
        self.Kp = servo_p
        self.Kd = servo_d
        self.error = 0
        self.last_error = 0
        self.target = 0
        self.output = 0
        self.output_limit = 3.8

    def calc_servo_pd(self, error, kp, kd):
        """
        Compute PD output for the given error.

        Args:
            error: Current position error (pixels from lane center).
            kp: Proportional gain (may vary by driving mode).
            kd: Derivative gain.

        Returns:
            Clamped output value for servo angle.
        """
        self.error = error
        self.Kp = kp
        self.Kd = kd
        self.output = self.Kp * self.error + self.Kd * (self.error - self.last_error)
        self.last_error = self.error

        if self.output > self.output_limit:
            self.output = self.output_limit
        elif self.output < -self.output_limit:
            self.output = -self.output_limit

        return self.output
