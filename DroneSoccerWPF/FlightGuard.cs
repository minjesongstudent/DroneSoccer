namespace DroneSoccerWPF;

/// <summary>
/// Last gate before a stick command reaches the Tello.
/// Keep this in step with flight_guard.py.
/// </summary>
public readonly record struct Stick(int Lr, int Fb, int Ud, int Yaw);

public readonly record struct GuardDecision(
    Stick Stick,
    bool ForwardBlocked,
    bool ClimbBlocked,
    bool DescendingToLimit);

public static class FlightGuard
{
    public static int ClampAxis(int value) => Math.Clamp(value, -100, 100);

    public static GuardDecision Apply(
        Stick desired,
        bool flying,
        int? heightCm,
        int maxHeightCm,
        bool obstacle)
    {
        var lr = ClampAxis(desired.Lr);
        var fb = ClampAxis(desired.Fb);
        var ud = ClampAxis(desired.Ud);
        var yaw = ClampAxis(desired.Yaw);

        var forwardBlocked = false;
        var climbBlocked = false;
        var descending = false;

        if (obstacle && fb > 0)
        {
            fb = 0;
            lr = 0;
            forwardBlocked = true;
        }

        if (flying && heightCm is not null && heightCm >= 0)
        {
            if (heightCm >= maxHeightCm && ud > 0)
            {
                ud = 0;
                climbBlocked = true;
            }

            if (heightCm > maxHeightCm + 8)
            {
                ud = Math.Min(ud, -30);
                descending = true;
                climbBlocked = true;
            }
        }

        return new GuardDecision(new Stick(lr, fb, ud, yaw), forwardBlocked, climbBlocked, descending);
    }
}
